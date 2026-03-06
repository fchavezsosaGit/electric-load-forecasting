# Changelog (SPEC-000)

All notable project planning and implementation changes should be documented here.
Entries are grouped by date and capability area so the history stays readable as the project grows.

## 2026-03-06

### Bootstrap and E2E Foundation Refresh

Scope:
- `config/pipeline.toml`
- `scripts/config.py`
- `scripts/bootstrap_env.py`
- `scripts/run_e2e.py`
- `setup.sh`
- `setup.ps1`
- `run_e2e.sh`
- `run_e2e.ps1`
- `.gitignore`
- `README.md`
- `docs/002_pipeline/pipeline.md`

What changed:
- Centralized generated artifact directories in TOML-backed `PATHS`:
  - `outputs_modeling_dir`
  - `outputs_performance_dir`
- Added a shared dependency/bootstrap helper so both shell families use the
  same environment installation and smoke-check path.
- Added a repo-level E2E runner that standardizes:
  - full pipeline rebuild
  - optional Stage-5 execution
  - notebook validation
  - pytest execution
- Added root wrapper scripts for local E2E.
- Tightened repository hygiene:
  - `.gitignore` extended for local editor files
  - hardcoded-path governance test expanded to cover repo-owned code, shell, TOML,
    notebook, and YAML files
- Made notebook execution dependencies explicit in `pyproject.toml` and corrected
  the `pyarrow` environment marker so supported environments remain eligible.

Why this was needed:
- The repo previously relied on good local state more than explicit bootstrap and
  verification entrypoints.
- Setup logic was duplicated across shell families, increasing drift risk.

## 2026-02-20

### Documentation and Verification Refresh (Latest)

Scope:
- `docs/000_governance/000_spec.md`
- `docs/000_governance/001_spec.md`
- `docs/change logs/000spec/changelog.md`
- `docs/change logs/001spec/changelog.md`
- `README.md`
- `docs/002_pipeline/pipeline.md`

What changed:
- Added a new SPEC-000 revision entry (`R6`) to capture the latest post-hardening
  verification snapshot.
- Updated stale verification totals from earlier snapshots to reflect current results:
  - `pytest -q` -> `98 passed`
  - `pyright run_pipeline.py scripts tests` -> `0 errors, 0 warnings`
  - `python run_pipeline.py --stage all` -> success
  - `python scripts/validate_notebooks.py` -> success (including silver mode matrix)
- Recorded notebook execution hardening details:
  - Windows event-loop policy fix in notebook validator for warning-free nbconvert runs.
  - QQ-plot scalar conversion hardening in raw notebook for Pylance type safety.
- Added synchronized verification snapshots to README and pipeline operations docs.

Why this was needed:
- Removes stale pass counts that could mislead reviewers and operators.
- Keeps source-of-truth specs, operational docs, and changelogs synchronized.
- Preserves auditability by documenting why the latest notebook/runtime fixes matter
  to repeatable, warning-free execution.

### Release Summary

- Completed Phase 1 and Phase 2 foundation hardening for ingestion, transformation, and modeling dataset preparation.
- Established a single governance source of truth and aligned repository structure to numbered conventions.
- Added robust runtime validation, deterministic stage behavior, and explicit operational logging.
- Expanded automated test coverage, added integration and notebook validation tests, and re-executed notebooks.
- Reorganized and normalized documentation so implementation, runbooks, and specs reference the same contracts.

### Governance and Planning

Scope:
- `docs/000_governance/000_spec.md`
- `docs/000_governance/001_spec.md`
- `changelog.md`
- `personal/improvements.md`

What changed:
- Set `docs/000_governance/000_spec.md` as the canonical implementation source of truth.
- Added `docs/000_governance/001_spec.md` (SPEC-01) -- five-phase, 18-step specification
  covering notebook configurability, expanded notebook development (QQ plots, PSD, STL, VIF,
  PACF, mutual information, interactive Plotly charts, data quality scorecards),
  in-notebook glossaries, and TOML configuration migration (`config/pipeline.toml`,
  `config/eda.toml`, `pyproject.toml`).
- Replaced over-scoped planning text in `personal/improvements.md` with a constrained two-phase execution plan.
- Removed residual pending checkbox markers in governance docs to prevent ambiguity about completion status.

Why this was needed:
- Keeps project decisions traceable to one authoritative spec.
- Reduces execution risk by sequencing foundational reliability work before advanced experimentation.

### Repository Structure and Naming Normalization

Scope:
- `data/`
- `scripts/`
- `docs/`
- `.gitignore`
- cross-references in README, notebooks, tests, and docs

What changed:
- Renamed data layer folders to numbered order:
  - `data/raw` -> `data/000_raw`
  - `data/bronze` -> `data/001_bronze`
  - `data/silver` -> `data/002_silver`
  - `data/gold` -> `data/003_gold`
  - `data/model` -> `data/004_model`
- Renamed core scripts to numbered order:
  - `scripts/01_raw_to_bronze.py` -> `scripts/000_raw_to_bronze.py`
  - `scripts/02_bronze_to_silver.py` -> `scripts/001_bronze_to_silver.py`
  - `scripts/03_silver_to_gold.py` -> `scripts/002_silver_to_gold.py`
  - `scripts/04_create_model_datasets.py` -> `scripts/003_create_model_datasets.py`
- Reorganized docs into numbered folders while keeping semantic markdown names:
  - `docs/000_governance`
  - `docs/001_architecture`
  - `docs/002_pipeline`
  - `docs/003_modeling`
  - `docs/004_reference`

Why this was needed:
- Enforces consistent repository navigation and ordering conventions.
- Prevents path drift and reduces maintenance errors across scripts/tests/docs.

### Pipeline Infrastructure Hardening

Scope:
- `scripts/config.py`
- `scripts/utils.py`
- `run_pipeline.py`

What changed:
- `scripts/config.py`
  - Added `validate_config()` runtime checks for feature window integrity, day class mapping integrity, resolution/alias integrity, feature-set schema validity, leakage prevention, and contiguous split ranges.
  - Added duplicate-column detection in `_build_silver_columns()`.
  - Added centralized `EDA_CONFIG`.
- `scripts/utils.py`
  - Added strict input validation for seasonal/time-of-day/slope helpers.
  - Added shared schema validation utility and documented edge-case behavior.
- `run_pipeline.py`
  - Added startup and dry-run config validation.
  - Added stage-level failure context with elapsed time.
  - Added stage controls (`--stage`, `--resolution`), `--dry-run`, `--verbose`, and file logging.

Why this was needed:
- Moves failure detection earlier and prevents silent data contract violations.
- Improves diagnosability and operational safety during iterative pipeline runs.

### Stage Implementations: Ingestion Through Model Datasets

Scope:
- `scripts/000_raw_to_bronze.py`
- `scripts/001_bronze_to_silver.py`
- `scripts/002_silver_to_gold.py`
- `scripts/003_create_model_datasets.py`

What changed:
- Raw to Bronze:
  - Added validation for missing keys, shape/dtype constraints, duplicate dates, and zero-day columns.
  - Added infinity normalization, all-NaN day warnings, physical-range warnings, monotonic timestamp checks, and schema enforcement.
  - Added detailed write logging (coverage, nulls, distribution, file size).
- Bronze to Silver:
  - Added defensive handling for empty/all-NaN inputs.
  - Added explicit resample semantics and monotonic index checks.
  - Added day-class integrity checks, inf handling, warm-up and drop-rate observability, and schema validation.
  - Added multi-resolution generation from config-defined defaults.
- Silver to Gold:
  - Added robust read/write error handling and required-column checks.
  - Added required-not-null filtering with per-column drop diagnostics and critical empty-output signaling.
  - Enforced schema and day-class validity post-filter.
- Gold to Model Datasets:
  - Added deterministic chronological train/validate/test splitting with overlap/order guards.
  - Added target leakage protections and explicit target inclusion.
  - Added split-level quality statistics and write protections.

Why this was needed:
- Converts ad hoc transformation behavior into repeatable, validated data contracts.
- Protects downstream modeling from malformed or ambiguous intermediate artifacts.

### Resolution Policy and Billing Alignment

Scope:
- `scripts/config.py`
- `scripts/001_bronze_to_silver.py`
- `scripts/002_silver_to_gold.py`
- `scripts/003_create_model_datasets.py`
- `run_pipeline.py`
- tests and docs

What changed:
- Expanded supported resolutions to:
  - `1s`, `5s`, `10s`, `30s`, `60s` (alias), `1min`, `5min`, `10min`, `15min`
- Updated defaults to include `15min`.
- Added alias normalization `60s -> 1min`.
- Updated output suffix mapping to include second-level and `15m` outputs.

Why this was needed:
- Supports both analysis flexibility and 15-minute operational settlement requirements.
- Reduces risk of late-stage reconciliation when billing intervals matter.

### Modeling Safety and Feature-Target Contract

Scope:
- `scripts/config.py`
- `scripts/003_create_model_datasets.py`
- `tests/stages/test_model_datasets.py`
- `tests/unit/test_config.py`
- related docs

What changed:
- Removed `avg_load` from all predictor feature sets and kept it as target only.
- Added config and runtime guards to block target leakage.
- Enforced split chronology checks (`train < validate < test`) and overlap rejection.
- Switched split NaN filtering to target-only and added per-feature null-rate logging.

Why this was needed:
- Eliminates leakage-prone configurations and protects experiment validity.
- Makes split behavior explicit, deterministic, and auditable.

### Notebook Standardization and Validation

Scope:
- `notebooks/000_raw_eda.ipynb`
- `notebooks/001_bronze_eda.ipynb`
- `notebooks/002_silver_eda.ipynb`
- `scripts/validate_notebooks.py`
- `tests/notebooks/test_validate_notebooks.py`
- `tests/notebooks/test_notebook_structure.py`

What changed:
- Refactored notebooks to use config-driven paths and clearer narrative structure.
- Added stronger EDA diagnostics across raw, bronze, and silver notebooks.
- Re-executed notebooks end to end so outputs and execution counts match current code/data.
- Added notebook smoke-run utility and notebook structure guard tests.

Why this was needed:
- Keeps notebook artifacts reproducible and review-ready.
- Prevents notebook drift from diverging from pipeline contracts.

### Test Coverage Expansion

Scope:
- `tests/conftest.py`
- `tests/unit/test_config.py`
- `tests/stages/test_raw_to_bronze.py`
- `tests/stages/test_bronze_to_silver.py`
- `tests/unit/test_feature_engineering.py`
- `tests/stages/test_silver_to_gold.py`
- `tests/stages/test_model_datasets.py`
- `tests/orchestration/test_run_pipeline.py`
- `tests/integration/test_integration.py`
- `tests/notebooks/test_validate_notebooks.py`

What changed:
- Added edge fixtures (`all_nan_bronze_df`, `single_row_bronze_df`, `single_day_bronze_df`, `empty_bronze_df`).
- Added orchestrator, integration, and notebook-validator test modules.
- Expanded validation coverage across config, stage error paths, schema guarantees, determinism, and split safety.

Why this was needed:
- Converts core data and orchestration assumptions into executable regression guards.
- Increases confidence for refactors and future automation.

### Documentation Reorganization and Professionalization

Scope:
- `README.md`
- `docs/002_pipeline/pipeline.md`
- `docs/002_pipeline/plan.md`
- `docs/003_modeling/feature_sets.md`
- `docs/003_modeling/hypothesis.md`
- `docs/003_modeling/mvmp.md`
- `docs/004_reference/glossary.md`
- architecture docs under `docs/001_architecture/`

What changed:
- Reworked README into a full operational runbook with architecture, commands, conventions, and doc index.
- Added/updated pipeline, plan, modeling, and glossary docs with consistent cross-references.
- Aligned documentation with current naming, schema, feature, and resolution contracts.
- Audited local markdown links and fixed reference drift.

Why this was needed:
- Keeps implementation and documentation synchronized for onboarding and review.
- Improves traceability from requirements to code to operations.

### Dependencies, Ignore Rules, and Generated Artifacts

Scope:
- dependency manifest (`pyproject.toml`)
- `.gitignore`
- generated parquet artifacts in `data/001_bronze`, `data/002_silver`, `data/003_gold`

What changed:
- Pinned and expanded runtime/test dependencies (`pyarrow`, `pytest`, `pytest-cov`, `jupyter`, `scikit-learn`).
- Corrected ignore pattern typo and expanded ignores for generated artifacts/logs.
- Regenerated bronze/silver/gold artifacts with hardened pipeline behavior.

Why this was needed:
- Improves environment reproducibility and reduces accidental artifact commits.
- Ensures sample outputs reflect current transformation logic.

### Documentation Hygiene, UTF-8 Safety, and Comment Standards

Scope:
- `README.md`
- `changelog.md`
- all `docs/**/*.md`
- Python modules and tests
- `personal/papers/generate_report3.py` (encoding cleanup)

What changed:
- Standardized module/class/function docstrings across Python code and tests.
- Re-audited docs against current script names, data paths, and conventions.
- Verified local markdown link integrity (128 links checked, no broken links).
- Completed UTF-8 safety pass and removed BOM/mojibake artifacts found during audit.

Why this was needed:
- Prevents documentation drift and encoding-related cross-platform failures.
- Keeps repository text assets professional, parseable, and maintainable.

### Verification Snapshot (Historical Early Pass)

- `pytest tests -q` -> `74 passed`
- `pytest tests --cov=scripts --cov=run_pipeline --cov-report=term -q` -> `74 passed`, coverage `80%`
- `python run_pipeline.py --dry-run` -> success

### Repository Audit Refresh

Scope:
- `config/pipeline.toml`
- `scripts/config.py`
- `scripts/000_raw_to_bronze.py`
- `scripts/001_bronze_to_silver.py`
- `scripts/003_create_model_datasets.py`
- `tests/unit/test_config.py`
- `README.md`
- `docs/001_architecture/000_overview/architecture.md`
- `docs/002_pipeline/pipeline.md`

What changed:
- Centralized remaining operational literals into TOML-backed config:
  - added `raw_contract.seconds_per_day` and `raw_contract.required_keys`
  - added `quality_thresholds.silver_nan_drop_warn_pct`
- Exported and validated new runtime constants in `scripts/config.py`:
  - `SECONDS_PER_DAY`, `MATLAB_REQUIRED_KEYS`, `SILVER_NAN_DROP_WARN_PCT`
- Removed hardcoded values from stage scripts:
  - raw ingestion now uses `SECONDS_PER_DAY` and `MATLAB_REQUIRED_KEYS`
  - silver warning threshold now uses `SILVER_NAN_DROP_WARN_PCT`
  - removed date-specific hardcoded operational note from model dataset script
- Expanded config tests for new TOML sections and round-trip validation.
- Updated architecture/pipeline/README docs to clarify config centralization and interim wrapper behavior.

Why this was needed:
- Completes config centralization for core operational thresholds/contracts.
- Reduces hidden stage behavior drift by making thresholds explicit and reviewable in TOML.
- Improved repository clarity during the transition period before wrapper removal.

### Repository Organization Cleanup (Root Duplicates and Dependency Source)

Scope:
- `README.md`
- dependency manifest (`pyproject.toml`)
- `pyproject.toml`
- `pyrightconfig.json`
- `docs/001_architecture/000_overview/architecture.md`
- `docs/004_reference/glossary.md`
- `utils.py` (root)

What changed:
- Removed repository-root `utils.py` compatibility wrapper to eliminate duplicate module surfaces.
- Kept `scripts/config.py` and `scripts/utils.py` as the only configuration/utility sources.
- Updated docs to remove stale root-wrapper references and document `scripts/` path bootstrapping.
- Consolidated dependency version definitions into `pyproject.toml` (`project.dependencies` + `dev` extras).
- Removed the legacy duplicate dependency manifest and kept `pyproject.toml` as the only dependency source.
- Added `pyrightconfig.json` with `extraPaths=["scripts"]` to keep static type checking stable after wrapper removal.

Why this was needed:
- Removes confusing duplicate module entrypoints that looked like accidental drift.
- Prevents dependency version divergence between two separate manifests.
- Preserves existing CLI/notebook/test runtime behavior while tightening repository structure.

### Repository Deep-Clean Follow-up (Notebook Issues, Dependency Cleanup, Test Structure)

Scope:
- `notebooks/000_raw_eda.ipynb`
- `notebooks/001_bronze_eda.ipynb`
- `notebooks/002_silver_eda.ipynb`
- `tests/notebooks/test_notebook_structure.py`
- `tests/notebooks/test_validate_notebooks.py`
- `tests/unit/*`
- `tests/stages/*`
- `tests/orchestration/*`
- `tests/integration/*`
- `README.md`
- `docs/004_reference/glossary.md`
- `docs/000_governance/000_spec.md`
- `docs/000_governance/001_spec.md`
- `docs/change logs/001spec/changelog.md`
- `personal/issues.md`

What changed:
- Resolved notebook static-analysis issues by switching notebook imports to `scripts.config` and `scripts.utils`.
- Replaced comment-only configuration notes with markdown cells that link directly to `config/eda.toml`, `config/pipeline.toml`, and `scripts/config.py`.
- Added typed `seaborn` style casts and explicit NumPy conversions in notebook cells flagged by Pylance diagnostics.
- Reorganized tests into domain subfolders:
  - `tests/unit`, `tests/stages`, `tests/orchestration`, `tests/integration`, `tests/notebooks`
- Updated notebook tests for new import style and nested test paths.
- Removed duplicate dependency file usage and normalized documentation/spec language to a single dependency source in `pyproject.toml`.
- Updated personal issue log with exact fixes and verification evidence.

Why this was needed:
- Converts notebook diagnostics from ambiguous IDE warnings into actionable, fixed code paths.
- Makes notebook configuration governance visible and discoverable via linked source files.
- Improves test suite maintainability by grouping tests by responsibility instead of a flat directory.
- Eliminates dependency manifest drift by enforcing one dependency source of truth.

### Root Tooling and Setup Bootstrap

Scope:
- `README.md`
- `setup.sh`
- `setup.ps1`

What changed:
- Added root bootstrap scripts:
  - `setup.sh` for Unix-like environments.
  - `setup.ps1` for PowerShell environments.
- Both scripts install dependencies from `pyproject.toml` via
  `pip install -e ".[dev]"` and support local virtual-environment workflows.
- Added README clarification that `pyproject.toml` and `pyrightconfig.json`
  intentionally stay at repository root (tool auto-discovery behavior), while
  `config/` holds runtime pipeline configuration TOML files.

Why this was needed:
- Provides one-command environment setup for both shell families.
- Prevents confusion between tooling manifests and runtime configuration files.

### Documentation Truth-Up and Professional Cleanup

Scope:
- `README.md`
- `docs/001_architecture/000_overview/architecture.md`
- `docs/002_pipeline/pipeline.md`
- `docs/002_pipeline/plan.md`
- `docs/004_reference/glossary.md`
- `docs/000_governance/001_spec.md`
- `docs/change logs/001spec/changelog.md`

What changed:
- Updated model-output naming references from `{res}`/`{resolution}` to the runtime-accurate
  `{suffix}` convention used by `scripts/003_create_model_datasets.py`.
- Clarified silver/gold output expectations in pipeline docs:
  default resolution outputs versus optional outputs generated only when explicitly requested.
- Refined README setup/testing sections for clearer onboarding:
  quick-start commands, `pytest` command normalization, and coverage command alignment.
- Cleaned glossary wording around import resolution behavior to match current runtime and notebook patterns.
- Updated verification snapshot text in SPEC-001 docs/changelog to the latest passing test count.
- Re-ran markdown link integrity checks to ensure all local links resolve.

Why this was needed:
- Aligns documentation with actual runtime behavior and file naming contracts.
- Reduces onboarding confusion by making defaults, options, and command expectations explicit.
- Improves overall documentation professionalism and consistency across root and `docs/`.

### Root Non-Markdown File Hardening (Purpose/Date Metadata)

Scope:
- `run_pipeline.py`
- `setup.sh`
- `setup.ps1`
- `pyproject.toml`
- `pyrightconfig.json`
- `.gitignore`
- `tests/notebooks/test_notebook_structure.py`

What changed:
- Added explicit purpose/date headers to root non-markdown operational files.
- Expanded `run_pipeline.py` module docstring with operational intent and review date.
- Added root-level metadata comments to bootstrap scripts and tooling manifests to clarify
  ownership and why each file exists.
- Added repository-level purpose/date header to `.gitignore`.
- Hardened notebook execution-guard test to treat metadata-based execution evidence as valid
  (covers nbconvert variants that omit numeric `execution_count`).

Why this was needed:
- Makes root operational files self-describing for maintainers and reviewers.
- Improves auditability by embedding review date and intent where changes are executed.
- Prevents false negatives in notebook execution checks caused by environment-specific metadata behavior.

### Pipeline Log Noise Elimination (Test Isolation)

Scope:
- `run_pipeline.py`
- `tests/orchestration/conftest.py`
- `tests/orchestration/test_run_pipeline.py`
- `README.md`
- `docs/002_pipeline/pipeline.md`

What changed:
- Added `ELF_PIPELINE_LOG_FILE` override support in `run_pipeline.py`:
  - unset -> default `logs/pipeline.log`
  - custom path -> write logs to that file
  - `off|none|disable|disabled|0|false` -> disable file logging (console only)
- Added orchestration test fixture to redirect pipeline file logging to per-test temp files.
- Extended orchestration tests to validate:
  - default file logging behavior
  - custom file-path override behavior
  - file-logging disable behavior
- Documented the log override in README and pipeline operations doc.

Why this was needed:
- Prevents intentional negative-path test failures from polluting repository runtime logs.
- Keeps `logs/pipeline.log` useful for actual operational diagnostics.
