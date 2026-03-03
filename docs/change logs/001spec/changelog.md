# Changelog (SPEC-001)

Source spec:
- `docs/000_governance/001_spec.md`

## 2026-02-20

### Verification Refresh

What changed:
- Re-ran full verification after final notebook and documentation updates.
- Confirmed all automated tests pass (`pytest -q`: `98 passed`).
- Confirmed notebook validator passes end-to-end:
  - baseline execution for `000_raw_eda.ipynb`, `001_bronze_eda.ipynb`, `002_silver_eda.ipynb`
  - silver matrix profiles (`default`, `all`, `custom`) with mixed AUTO settings.
- Confirmed static analysis passes for runtime code paths:
  - `pyright run_pipeline.py scripts tests` -> `0 errors, 0 warnings`.
- Hardened notebook execution/runtime safety:
  - updated `scripts/validate_notebooks.py` to execute nbconvert through Python with
    `WindowsSelectorEventLoopPolicy` on Windows to eliminate prior `zmq` runtime warnings.
  - tightened QQ-plot scalar conversion in `notebooks/000_raw_eda.ipynb` to resolve
    Pylance `reportArgumentType` diagnostics.
- Confirmed no unresolved `TODO`/unchecked checklist markers in tracked docs/code.

Why this was needed:
- Provides an auditable final health check proving implementation and documentation are synchronized.
- Reduces risk of regressions before handoff by validating both tests and notebook execution artifacts.

### Release Summary

- Implemented all five SPEC-001 phases end-to-end:
  - Phase 1: configuration infrastructure and self-optimizing utilities.
  - Phase 2: standardized notebook configuration cells and glossary sections.
  - Phase 3: expanded raw, bronze, and silver notebook analyses.
  - Phase 4: TOML migration and `pyproject.toml` bootstrap.
  - Phase 5: validation, documentation updates, and changelog consolidation.
- Completed execution validation:
  - `python scripts/validate_notebooks.py` passed (including silver mode/profile matrix).
  - `pytest -q` passed.

### Phase 1: Configuration Infrastructure

Files:
- `scripts/config.py`
- `scripts/utils.py`
- `tests/unit/test_config.py`
- `tests/unit/test_feature_engineering.py`

What changed:
- Extended `EDA_CONFIG` to include all notebook-governed visualization and analysis defaults.
- Added resolution mode infrastructure:
  - `EDA_RESOLUTION_MODES`, `EDA_DEFAULT_RESOLUTION_MODE`
  - `resolve_eda_resolutions()`, `resolve_resolution_suffix()`
  - `get_silver_path()`, `get_gold_path()`
- Added self-optimizing helpers in `scripts/utils.py`:
  - `optimal_bin_count()`
  - `adaptive_outlier_threshold()`
  - `optimal_acf_depth()`
- Expanded test coverage for new config exports and utility behavior (normal, edge, and failure paths).

Why this was needed:
- Centralized notebook parameters to prevent cell-level drift.
- Enabled reusable, validated resolution selection logic for all notebook consumers.
- Replaced fixed heuristics with data-adaptive defaults while retaining deterministic fallbacks.

### Phase 2 and Phase 3: Notebook Refactor and Expanded EDA

Files:
- `notebooks/000_raw_eda.ipynb`
- `notebooks/001_bronze_eda.ipynb`
- `notebooks/002_silver_eda.ipynb`
- `scripts/validate_notebooks.py`
- `tests/notebooks/test_validate_notebooks.py`

What changed:
- Added standardized configuration cells immediately after notebook imports.
- Removed hardcoded analysis/visualization constants from downstream cells and sourced settings from `EDA_CONFIG`.
- Added required glossary sections at notebook end with layer-specific terms.
- Expanded analyses:
  - Raw: QQ plot, PSD, day-class transition matrix, stratified box plots, interactive drill-down, scorecard.
  - Bronze: STL decomposition, transition analysis, ramp-rate diagnostics, interactive overlay, cross-day correlation, scorecard.
  - Silver: MI/Pearson ranking, VIF, PACF with ACF, cross-resolution fidelity, stratified target analysis, rolling comparisons, interactive correlation explorer, scorecard.
- Hardened silver feature diagnostics to handle non-numeric, constant, and NaN/inf feature inputs safely.
- Added env-driven notebook overrides (`ELF_NB_*`) so validation can exercise mode and flag combinations without manual edits.
- Updated notebook validator to execute silver in:
  - `default` mode with auto settings,
  - `all` mode with fixed-parameter fallbacks,
  - `custom` mode (`5min,15min`) with mixed auto/fixed settings.

Why this was needed:
- Made notebooks reproducible, parameterized, and review-ready for advisor walkthroughs.
- Added the diagnostics required for model-facing interpretation and feature governance.
- Ensured automated validation covers the exact operating modes defined by the spec.

### Phase 4: TOML Configuration Migration

Files:
- `config/pipeline.toml`
- `config/eda.toml`
- `scripts/config.py`
- `pyproject.toml`
- dependency declarations in `pyproject.toml`

What changed:
- Migrated declarative config into TOML:
  - pipeline contracts and resolution policy in `config/pipeline.toml`
  - notebook defaults in `config/eda.toml`
- Refactored `scripts/config.py` to load TOML via `tomllib`, normalize types, and preserve existing public exports.
- Preserved computed values in Python (`SILVER_COLUMNS`, schemas, full feature set, runtime validation).
- Added explicit missing-config error handling with informative `FileNotFoundError`.
- Added project metadata/tooling bootstrap in `pyproject.toml`.
- Added `statsmodels>=0.14,<1.0` to `pyproject.toml` dependencies.

Why this was needed:
- Separated declarative settings from runtime logic.
- Improved reviewability/editability for non-code configuration changes.
- Preserved backward compatibility for existing imports while modernizing config storage.

### Phase 5: Validation and Documentation

Files:
- `docs/001_architecture/000_overview/architecture.md`
- `docs/002_pipeline/pipeline.md`
- `docs/004_reference/glossary.md`
- `README.md`
- `changelog.md`
- `docs/000_governance/001_spec.md`

What changed:
- Updated architecture docs with TOML-backed configuration flow and notebook resolution mode behavior.
- Updated pipeline docs with notebook configuration governance, env override controls, and TOML layout.
- Expanded glossary with SPEC-001 terms (configuration cell, self-optimizing parameter, TOML, MI, VIF, PACF, PSD, STL, scorecard, etc.).
- Updated README project structure and setup to include `config/` TOML files.
- Updated root changelog index to mark SPEC-001 active and point to this file for implementation details.
- Updated SPEC-001 status/progress in the governing specification document.

Why this was needed:
- Kept documentation synchronized with the implemented architecture and operational workflows.
- Ensured governance records remain auditable and consistent with the source-of-truth spec model.
