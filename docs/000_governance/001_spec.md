# SPEC-01: Notebook Development, Configurability, and Configuration Migration

| Field            | Value                                                        |
|------------------|--------------------------------------------------------------|
| Project          | Daily Electric Load Forecasting                              |
| Specification ID | SPEC-01                                                      |
| Status           | Implemented (validated 2026-02-20)                           |
| Created          | 2026-02-20                                                   |
| Last Updated     | 2026-02-20                                                   |
| Authors          | Spencer Hoyle, Sean He, Frank Chavezsosa                     |
| Advisor          | Prof. Raymond de Callafon                                    |
| Depends on       | SPEC-00 (completed)                                          |

## Implementation status

As of 2026-02-20, all phases in this specification have been implemented and
validated, including a final verification refresh after notebook/runtime hardening.

Validation snapshot:
- `python run_pipeline.py --stage all` passed.
- `python scripts/validate_notebooks.py` passed, including silver resolution modes
  (`default`, `all`, `custom`) and mixed AUTO-flag profiles, with no warning output.
- `pytest -q` passed with full summary: `98 passed`.
- `pyright run_pipeline.py scripts tests` passed with `0 errors, 0 warnings`.
- Notebook/static-analysis hardening was revalidated:
  - `scripts/validate_notebooks.py` now executes nbconvert with a Windows selector
    event loop policy to avoid the prior `zmq` runtime warning.
  - `notebooks/000_raw_eda.ipynb` QQ-plot scalar conversion was tightened to satisfy
    Pylance `reportArgumentType` diagnostics.

Implementation details are recorded in the spec-specific changelog:
- [docs/change logs/001spec/changelog.md](../change%20logs/001spec/changelog.md)

## Source of truth declaration

For notebook development, configurability, and TOML migration work, this file is the
canonical implementation specification. If any planning text conflicts with implementation
details in code or documentation, this file and the matching entries in the spec-specific
changelog at [docs/change logs/001spec/changelog.md](../change%20logs/001spec/changelog.md)
take precedence. The root [changelog.md](../../changelog.md) serves as an index pointing
to spec-specific changelogs.

## Why this specification exists

The three EDA notebooks (`000_raw_eda.ipynb`, `001_bronze_eda.ipynb`, `002_silver_eda.ipynb`)
currently contain hardcoded parameters scattered throughout their cell bodies: figure sizes,
histogram bin counts, z-score thresholds, outlier criteria, resolution selections, percentile
lists, and analysis window sizes. This makes the notebooks rigid and forces manual edits
whenever the team wants to explore a different resolution or adjust a visualization parameter.

Beyond configuration, the notebooks need broader development as analysis and presentation
artifacts. They lack visualizations and analyses that are standard for a time-series
forecasting project: seasonal decomposition, transition behavior, power spectral density,
QQ normality checks, interactive drill-downs, and cross-layer data quality scorecards. The
notebooks are also not self-documenting -- a reader unfamiliar with the project must
cross-reference multiple markdown files to understand the terminology and methods used.

Finally, the project's configuration layer (`scripts/config.py`) embeds declarative data
(paths, thresholds, feature lists, schemas) inside executable Python code. This couples
configuration values to the Python runtime and makes it difficult for non-developers to
review or edit settings. TOML is the standard format for declarative project configuration
in the Python ecosystem, and migrating to it improves readability, tooling, and separation
of concerns.

This specification addresses five problems:

1. **Configuration fragmentation.** Thresholds and parameters are embedded in code cells
   rather than declared at the top of each notebook or sourced from a central location.
   A teammate changing one threshold must hunt through cells to find it.

2. **Resolution inflexibility.** The silver EDA notebook loads only `1m`, `5m`, and `10m`
   resolutions with hardcoded file paths. There is no mechanism to select which resolutions
   to analyze, run all available resolutions, or specify a custom subset.

3. **Hardened parameters.** Numeric thresholds (z-score cutoffs, bin counts, lag depths) are
   fixed constants with no documentation of how they were chosen. Parameters that could be
   derived from the data itself are instead set to arbitrary values.

4. **Incomplete notebook development.** The notebooks lack standard time-series diagnostics
   (decomposition, spectral analysis, normality testing, transition behavior) and
   cross-layer quality summaries that are expected in a professional forecasting project.

5. **Configuration-code coupling.** All configuration lives in Python source files, mixing
   declarative settings with runtime logic. Moving to TOML separates what is configured
   from how it is used.

Primary goals:
- Make every notebook fully configurable from a single cell at the top.
- Develop notebooks with deeper visualizations and analyses that support modeling decisions.
- Add in-notebook glossaries so each notebook is self-contained and presentation-ready.
- Migrate declarative configuration from Python to TOML.

## Scope boundary

This specification covers development of the three existing notebooks only:

- `notebooks/000_raw_eda.ipynb`
- `notebooks/001_bronze_eda.ipynb`
- `notebooks/002_silver_eda.ipynb`

**No work beyond the silver layer is in scope.** Gold-layer notebooks, model-layer notebooks,
and any other new notebooks are explicitly excluded. If during implementation a need arises
for an additional notebook (e.g., gold EDA, model evaluation, or cross-layer summary), the
implementer must ask the team before creating it. The team will evaluate whether the new
notebook belongs in this spec, a future spec, or is unnecessary.

This boundary exists because the raw-through-silver notebooks are the foundation for all
downstream analysis. They must be solid, configurable, and well-documented before the team
invests in additional notebooks.

## Platform compatibility

This project runs on **ARM64 (Apple Silicon)** and **macOS**. All libraries used in
notebooks must have pre-built wheels or pure-Python fallbacks for both architectures. If a
library does not have ARM64 macOS support, the implementer must flag it to the team before
proceeding -- do not silently substitute or skip.

### Library compatibility audit

The following table documents ARM64 macOS wheel availability for every library used or
proposed by this specification. Status is based on the latest available releases as of
the specification date. If a library has since added ARM64 support that is not reflected
here, the implementer should verify and update this table.

| Library | Used for | ARM64 macOS wheels | Notes |
|---------|----------|-------------------|-------|
| `numpy` >=1.24 | Array operations | Yes (native since 1.21) | No issues expected |
| `pandas` >=2.0 | DataFrames | Yes (native since 1.4) | No issues expected |
| `scipy` >=1.11 | PSD, QQ plots, statistics | Yes (native since 1.9) | No issues expected |
| `matplotlib` >=3.7 | Static plots | Yes (native since 3.5) | No issues expected |
| `seaborn` >=0.12 | Statistical plots | Yes (pure Python) | No issues expected |
| `plotly` >=5.15 | Interactive charts | Yes (pure Python + JS) | No issues expected |
| `scikit-learn` >=1.3 | Mutual information | Yes (native since 1.1) | No issues expected |
| `statsmodels` >=0.14 | STL, VIF, PACF, ADF | Yes (native since 0.14) | Verify wheel exists for exact version |
| `pyarrow` >=14.0 | Parquet I/O | Yes (native since 10.0) | No issues expected |
| `tomllib` | TOML parsing | Yes (stdlib since 3.11) | No external dependency |
| `jupyter` >=1.0 | Notebook execution | Yes | No issues expected |
| `pytest` >=7.0 | Testing | Yes (pure Python) | No issues expected |

**`statsmodels`** is the only new runtime dependency introduced by this spec (for STL
decomposition, VIF, PACF, and ADF tests). It must be declared in `pyproject.toml`
with a version that has confirmed ARM64 macOS wheels. As of this writing,
`statsmodels>=0.14,<1.0` is recommended.

### Compatibility escalation protocol

If during implementation any library:
- Does not install cleanly on ARM64 macOS, or
- Requires building from source with no pre-built wheel, or
- Has a known incompatibility with the project's Python version (3.11+),

the implementer must:
1. Stop and notify the team with the library name, version, and error message.
2. Propose alternatives (different library, different approach, local build).
3. Wait for team decision before proceeding.

Do not silently work around compatibility issues by downgrading, patching, or skipping
analyses.

## Design principles

1. **Top-of-notebook configuration.** Every adjustable parameter is declared in a clearly
   labeled configuration cell immediately after imports. No parameter tuning requires
   editing any cell below the configuration cell.

2. **Central source of truth.** Default values for all EDA parameters live in a TOML
   configuration file (initially `scripts/config.py`, migrated to `pyproject.toml` or
   `config/eda.toml`). Notebooks import these defaults and may override them in the
   configuration cell when needed for a specific analysis.

3. **No hardened parameters.** A parameter is "hardened" when it is a fixed numeric literal
   with no documented rationale and no mechanism for adjustment. Every parameter must be
   either:
   - **Centrally defined** in the configuration source with a documented default, or
   - **Self-optimized** (computed from data characteristics at runtime using a documented
     algorithm).

4. **Resolution as a first-class control.** Each notebook that works with resolution-specific
   data provides a resolution selector in the configuration cell supporting three modes:
   run all available resolutions, run default resolutions, or run a user-specified list.

5. **Backward compatibility.** Running a notebook with no configuration changes produces
   output equivalent to the current notebooks. The configuration cell simply makes the
   implicit parameters explicit and adjustable.

6. **Self-documenting notebooks.** Each notebook ends with a markdown glossary cell that
   defines every term, metric, and method used in that notebook. A reader should be able
   to understand the notebook without opening any external document.

7. **TOML for data, Python for logic.** Declarative configuration (paths, thresholds,
   feature lists, schemas, color palettes) belongs in TOML. Computed configuration
   (column construction, validation functions, self-optimizing algorithms) remains in Python.

## Initial state snapshot

### Current hardcoded parameters by notebook

**`000_raw_eda.ipynb`** (14 cells):

| Parameter | Current Value | Location | Category |
|-----------|---------------|----------|----------|
| Figure size | `(12, 5)` via `plt.rcParams` | Cell 2 | Visualization |
| Heatmap figure size | `(14, 8)` | Cell 7 | Visualization |
| Histogram bins | `120` | Cell 9 | Analysis |
| Outlier threshold | `3` (sigma) | Cell 9 | Analysis |
| Profile figure size | `(14, 6)` | Cell 11 | Visualization |
| Minute reshape factor | `60` (hardcoded 1-min aggregation) | Cell 11 | Analysis |
| Label count for legend | `3` (first 3 days labeled) | Cell 11 | Visualization |

**`001_bronze_eda.ipynb`** (15 cells):

| Parameter | Current Value | Location | Category |
|-----------|---------------|----------|----------|
| Figure size | `(12, 5)` via `plt.rcParams` | Cell 2 | Visualization |
| Heatmap figure size | `(14, 8)` | Cell 6 | Visualization |
| Z-score threshold | `2.5` | Cell 8 | Analysis |
| Overlay resample frequency | `'5min'` | Cell 10 | Analysis |
| Overlay figure size | `(14, 6)` | Cell 10 | Visualization |
| Percentiles in describe | `[0.01, 0.05, 0.5, 0.95, 0.99]` | Cell 12 | Analysis |
| Zero-run threshold | `300` seconds (5 minutes) | Cell 14 | Analysis |
| Day-class colors | `{'full': '#2ca02c', 'half': '#1f77b4', 'none': '#ff7f0e'}` | Cell 10 | Visualization |

**`002_silver_eda.ipynb`** (15 cells):

| Parameter | Current Value | Location | Category |
|-----------|---------------|----------|----------|
| Figure size | `(12, 5)` via `plt.rcParams` | Cell 2 | Visualization |
| Resolutions loaded | `1m`, `5m`, `10m` (hardcoded paths) | Cell 2 | Resolution |
| Heatmap figure size | `(16, 12)` | Cell 6 | Visualization |
| Top correlations count | `20` | Cell 6 | Analysis |
| Feature distribution bins | `40` | Cell 10 | Analysis |
| Distribution figure size | `(16, 8)` | Cell 10 | Visualization |
| Autocorrelation max lag | `240` | Cell 12 | Analysis |
| Autocorrelation figure size | `(14, 4)` | Cell 12 | Visualization |
| Hourly profile figure size | `(12, 5)` | Cell 14 | Visualization |
| Features to plot | `['avg_load', 'workday', ...]` (manual list) | Cell 10 | Analysis |

### Current `EDA_CONFIG` in `scripts/config.py`

The following centralized defaults already exist:

| Key | Value | Used by notebooks |
|-----|-------|-------------------|
| `zscore_threshold` | `3.0` | Not yet imported by any notebook |
| `histogram_bins` | `50` | Not yet imported by any notebook |
| `figure_size` | `(14, 6)` | Not yet imported by any notebook |
| `physical_load_max_watts` | `100000.0` | Not yet imported by any notebook |
| `physical_load_min_watts` | `0.0` | Not yet imported by any notebook |
| `correlation_high_threshold` | `0.95` | Not yet imported by any notebook |
| `percentiles` | `[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]` | Not yet imported by any notebook |

None of these centralized values are currently imported by the notebooks.

### Missing EDA analyses by notebook

**`000_raw_eda.ipynb`** -- missing:
- QQ plot for normality assessment of raw load distribution.
- Power spectral density to identify dominant periodicities.
- Day-class stratified box plots comparing load distributions.
- Transition matrix between consecutive day classes.
- Interactive time-series drill-down for anomaly inspection.

**`001_bronze_eda.ipynb`** -- missing:
- Seasonal/STL decomposition of daily mean load.
- Day-class transition visualization (what day type follows what).
- Load ramp-rate distribution (how fast does load change).
- Interactive per-day overlay with hover tooltips.
- Cross-day correlation matrix (do similar days cluster).
- Data quality scorecard (summary pass/fail table).

**`002_silver_eda.ipynb`** -- missing:
- Feature importance preview (mutual information or correlation ranking).
- VIF (Variance Inflation Factor) multicollinearity quantification.
- Per-resolution rolling statistics comparison.
- Lag selection diagnostic (partial autocorrelation for lag guidance).
- Cross-resolution signal fidelity comparison (information loss quantification).
- Stratified target analysis (avg_load by day_class, hour, season).
- Interactive correlation explorer with resolution toggle.
- Data quality scorecard for silver outputs.

### No notebook glossaries exist

None of the three notebooks contain a glossary or term-definition section. Readers must
cross-reference [glossary.md](../004_reference/glossary.md), layer documentation, and
feature set definitions to understand terminology.

---

## Phase 1: Configuration Infrastructure

Goal: Extend `scripts/config.py` with the resolution selection mechanism, self-optimizing
parameter functions, and visualization defaults so notebooks have a complete configuration
source to import from.

Exit criteria:
- `EDA_CONFIG` contains all parameters currently hardcoded in notebooks.
- Resolution selection utilities are importable and tested.
- Self-optimizing parameter functions exist for bin count, outlier threshold, and
  autocorrelation depth.

---

### Step 1. Extend `EDA_CONFIG` with complete notebook parameters

Rationale: `EDA_CONFIG` exists but covers only 7 of the 25+ parameters currently hardcoded
across the three notebooks. The remaining parameters (figure sizes per plot type, day-class
color palette, zero-run threshold, autocorrelation depth, top-correlation count, and
distribution feature lists) have no central definition. Extending `EDA_CONFIG` provides a
single import target for all notebook configuration cells.

Tasks:

Visualization defaults (add to `EDA_CONFIG`):
- Add `figure_size_wide`: `(14, 8)` -- for heatmaps and full-width plots.
- Add `figure_size_compact`: `(14, 4)` -- for single-axis line plots (autocorrelation).
- Add `figure_size_grid`: `(16, 8)` -- for multi-panel subplot grids.
- Add `figure_size_correlation`: `(16, 12)` -- for full correlation heatmaps.
- Add `day_class_colors`: `{"full": "#2ca02c", "half": "#1f77b4", "none": "#ff7f0e"}` -- standardized color palette.
- Add `seaborn_style`: `"whitegrid"` -- plot theme.

Analysis defaults (add to `EDA_CONFIG`):
- Add `top_correlations_count`: `20` -- number of highest correlations to display.
- Add `zero_run_threshold_seconds`: `300` -- minimum sustained zero-load run to flag.
- Add `overlay_resample_frequency`: `"5min"` -- resample frequency for day-overlay plots.
- Add `legend_max_labels`: `3` -- maximum individual labels in dense overlay plots.
- Add `distribution_features`: `["avg_load", "workday", "day_of_week", "hour", "season", "time_of_day"]` -- features for distribution grid plots.

Update `EDAConfig` TypedDict:
- Add type annotations for all new keys.
- Ensure all existing and new keys have documented defaults.

Do not remove or rename any existing `EDA_CONFIG` keys -- maintain backward compatibility.

Acceptance criteria:
- `EDA_CONFIG` contains entries for every parameter currently hardcoded in the three notebooks.
- `EDAConfig` TypedDict matches the dictionary keys exactly.
- Importing `from scripts.config import EDA_CONFIG` provides access to all visualization and analysis defaults.
- Existing `EDA_CONFIG` keys and values are unchanged.

---

### Step 2. Add resolution selection infrastructure

Rationale: The silver EDA notebook loads three hardcoded resolutions (`1m`, `5m`, `10m`),
excluding `15m` (a default pipeline resolution) and all sub-minute resolutions. The team needs
to select resolutions flexibly: run all available, run the pipeline defaults, or specify an
explicit list. This selection mechanism must be importable so every resolution-aware notebook
uses the same logic.

Tasks:

Add to `scripts/config.py`:
- Add `EDA_RESOLUTION_MODES` constant documenting the three supported modes:
  - `"all"`: use `SUPPORTED_RESOLUTIONS` (all 8 resolutions).
  - `"default"`: use `DEFAULT_RESOLUTIONS` (`1min`, `5min`, `10min`, `15min`).
  - `"custom"`: use a user-provided list validated against `SUPPORTED_RESOLUTIONS`.
- Add `EDA_DEFAULT_RESOLUTION_MODE`: `"default"` -- the mode used when notebooks run without
  user override.
- Add a `resolve_eda_resolutions(mode, custom_list=None)` function:
  - If `mode == "all"`: return `list(SUPPORTED_RESOLUTIONS)`.
  - If `mode == "default"`: return `list(DEFAULT_RESOLUTIONS)`.
  - If `mode == "custom"`: validate every entry in `custom_list` against
    `SUPPORTED_RESOLUTIONS` (resolving aliases via `RESOLUTION_ALIASES`), raise `ValueError`
    for unknown resolutions, return the validated list.
  - If `mode` is not one of the three recognized values: raise `ValueError`.
- Add a `resolve_resolution_suffix(resolution)` helper that returns the file suffix for a
  resolution string (e.g., `"5min"` returns `"5m"`), resolving aliases first. Raise
  `ValueError` for unknown resolutions.
- Add a `get_silver_path(resolution)` helper that returns
  `PATHS["silver_dir"] / f"power_load_{suffix}.parquet"` for a given resolution.
- Add a `get_gold_path(resolution)` helper that returns
  `PATHS["gold_dir"] / f"power_load_{suffix}_all_features.parquet"` for a given resolution.

Acceptance criteria:
- `resolve_eda_resolutions("all")` returns all 8 supported resolutions.
- `resolve_eda_resolutions("default")` returns the 4 default resolutions.
- `resolve_eda_resolutions("custom", ["5min", "15min"])` returns `["5min", "15min"]`.
- `resolve_eda_resolutions("custom", ["60s"])` resolves alias and returns `["1min"]`.
- `resolve_eda_resolutions("custom", ["2min"])` raises `ValueError`.
- `resolve_eda_resolutions("invalid_mode")` raises `ValueError`.
- `get_silver_path("5min")` returns the correct `Path` object.
- `get_gold_path("1min")` returns the correct `Path` object.

---

### Step 3. Add self-optimizing parameter functions

Rationale: Several analysis parameters (histogram bin count, outlier threshold,
autocorrelation depth) are currently arbitrary constants. These parameters have well-known
data-driven selection methods that produce better results than fixed defaults. By providing
functions that compute optimal values from data characteristics, the notebooks automatically
adapt to different datasets or resolutions without manual tuning.

Tasks:

Add to `scripts/utils.py` (or a new `scripts/eda_utils.py` if preferred):

Optimal bin count:
- Add `optimal_bin_count(data, method="fd")` function.
  - `method="fd"` (default): Freedman-Diaconis rule.
    `bin_width = 2 * IQR(data) * n^(-1/3)`, `bins = ceil((max - min) / bin_width)`.
  - `method="sturges"`: Sturges' rule. `bins = ceil(log2(n)) + 1`.
  - `method="sqrt"`: Square root rule. `bins = ceil(sqrt(n))`.
  - Accept a `min_bins` (default 10) and `max_bins` (default 300) to clamp output.
  - Handle edge cases: constant data returns `min_bins`, empty data returns `min_bins`,
    all-NaN data returns `min_bins`.
  - Drop NaN values before computation.
- Document: the Freedman-Diaconis rule adapts to data spread and sample size, producing
  fewer bins for smooth distributions and more bins for multimodal ones.

Adaptive outlier threshold:
- Add `adaptive_outlier_threshold(data, method="iqr")` function.
  - `method="iqr"` (default): IQR-based. Outlier if value < Q1 - 1.5*IQR or
    value > Q3 + 1.5*IQR. Return the multiplier and bounds.
  - `method="zscore"`: Return `EDA_CONFIG["zscore_threshold"]` (central default).
  - `method="mad"`: Median Absolute Deviation. Threshold =
    `median + k * MAD` where `k = 3.0`. More robust to skewed distributions than z-score.
  - Return a named tuple or dictionary: `{"lower": float, "upper": float, "method": str}`.
  - Drop NaN values before computation.
- Document: for skewed load distributions (common in commercial facilities), IQR and MAD
  methods produce more meaningful outlier bounds than symmetric z-score thresholds.

Autocorrelation depth:
- Add `optimal_acf_depth(series, significance_level=0.05)` function.
  - Compute autocorrelation for increasing lags until ACF drops below the significance
    bound `1.96 / sqrt(n)` for `k` consecutive lags (default `k=5`).
  - Return the lag at which sustained insignificance begins, clamped to
    `[min_depth, max_depth]` where `min_depth=10` and `max_depth=2000`.
  - If the series is too short or all-NaN, return `min_depth`.
- Document: this avoids both truncating informative lags (current `max_lag=240` may be too
  low at some resolutions) and wasting computation on insignificant lags.

Acceptance criteria:
- `optimal_bin_count` returns Freedman-Diaconis bins for a normal sample and clamps
  within `[min_bins, max_bins]`.
- `optimal_bin_count` returns `min_bins` for constant or empty data.
- `adaptive_outlier_threshold` with `method="iqr"` returns bounds consistent with
  `Q1 - 1.5*IQR` and `Q3 + 1.5*IQR`.
- `adaptive_outlier_threshold` with `method="zscore"` returns bounds based on
  `EDA_CONFIG["zscore_threshold"]`.
- `optimal_acf_depth` returns a larger depth for highly autocorrelated series and
  `min_depth` for white noise.
- All functions handle NaN, empty, and constant input without raising.
- All functions have docstrings documenting the algorithm, parameters, and return type.

---

### Step 4. Add tests for new configuration and utility functions

Rationale: Steps 1-3 introduce new importable functions and configuration that will be used
by all three notebooks. These must be tested before notebooks depend on them.

Tasks:

Add to `tests/unit/test_config.py`:
- Test that all new `EDA_CONFIG` keys exist and have the expected types.
- Test that `resolve_eda_resolutions("all")` returns all supported resolutions.
- Test that `resolve_eda_resolutions("default")` returns default resolutions.
- Test that `resolve_eda_resolutions("custom", [...])` validates and resolves aliases.
- Test that `resolve_eda_resolutions("custom", ["invalid"])` raises `ValueError`.
- Test that `resolve_eda_resolutions("bad_mode")` raises `ValueError`.
- Test that `get_silver_path` and `get_gold_path` return correct paths for each default
  resolution.

Add to `tests/unit/test_feature_engineering.py` (or a new `tests/unit/test_eda_utils.py`):
- Test `optimal_bin_count` with normal data, constant data, empty array, all-NaN array.
- Test `optimal_bin_count` respects `min_bins` and `max_bins` clamping.
- Test `optimal_bin_count` with each method (`"fd"`, `"sturges"`, `"sqrt"`).
- Test `adaptive_outlier_threshold` with symmetric data (IQR method).
- Test `adaptive_outlier_threshold` with skewed data (MAD method).
- Test `adaptive_outlier_threshold` with `method="zscore"` returns config default.
- Test `optimal_acf_depth` with highly autocorrelated series (AR(1) process).
- Test `optimal_acf_depth` with white noise returns `min_depth`.
- Test `optimal_acf_depth` with all-NaN series returns `min_depth`.

Acceptance criteria:
- All new tests pass.
- No existing tests are broken by the config extension.
- Coverage for new functions is at least 90%.

---

## Phase 2: Notebook Configuration Cells and Glossaries

Goal: Refactor each notebook to declare all configuration in a single top-of-notebook cell,
import defaults from `scripts/config.py`, eliminate all hardcoded parameters from analysis
cells, and add a self-contained glossary at the end of every notebook.

Exit criteria:
- Every notebook has a clearly labeled configuration cell immediately after the import cell.
- No numeric literals for thresholds, sizes, bins, or lag depths appear below the
  configuration cell.
- Resolution selection is controllable from the configuration cell in every
  resolution-aware notebook.
- Every notebook ends with a markdown glossary defining all terms used in that notebook.
- Running each notebook with default configuration produces output equivalent to the
  current notebooks.

---

### Step 5. Define the standard configuration cell template

Rationale: All three notebooks should follow a consistent configuration pattern so the team
knows exactly where to look for adjustable parameters in any notebook. A shared template
reduces cognitive overhead and prevents configuration drift between notebooks.

Tasks:

Define a standard two-cell pattern for the top of every notebook:

Cell 1 -- Imports (already exists, minor cleanup):
- Project root discovery and `sys.path` setup (existing pattern).
- `from scripts.config import PATHS, EDA_CONFIG, SUPPORTED_RESOLUTIONS, DEFAULT_RESOLUTIONS`
- `from scripts.config import resolve_eda_resolutions, get_silver_path, get_gold_path`
  (for resolution-aware notebooks only)
- Standard library imports (`numpy`, `pandas`, `matplotlib`, `seaborn`).
- Self-optimizing utility imports:
  `from scripts.utils import optimal_bin_count, adaptive_outlier_threshold, optimal_acf_depth`
  (only import what the specific notebook needs).

Cell 2 -- Configuration (new cell, inserted after imports):
- Begin with a markdown-style comment block:
  ```python
  # === NOTEBOOK CONFIGURATION ===
  # Adjust parameters below to control analysis behavior.
  # Defaults are imported from config.py EDA_CONFIG.
  # To restore defaults, delete overrides and re-run this cell.
  ```
- Resolution selection (resolution-aware notebooks only):
  ```python
  RESOLUTION_MODE = "default"        # "all", "default", or "custom"
  CUSTOM_RESOLUTIONS = ["5min", "15min"]  # used only when RESOLUTION_MODE == "custom"
  RESOLUTIONS = resolve_eda_resolutions(RESOLUTION_MODE, CUSTOM_RESOLUTIONS)
  ```
- Visualization parameters (all notebooks):
  ```python
  FIGURE_SIZE = EDA_CONFIG["figure_size"]
  FIGURE_SIZE_WIDE = EDA_CONFIG["figure_size_wide"]
  FIGURE_SIZE_COMPACT = EDA_CONFIG["figure_size_compact"]
  FIGURE_SIZE_GRID = EDA_CONFIG["figure_size_grid"]
  FIGURE_SIZE_CORRELATION = EDA_CONFIG["figure_size_correlation"]
  DAY_CLASS_COLORS = EDA_CONFIG["day_class_colors"]
  SEABORN_STYLE = EDA_CONFIG["seaborn_style"]
  ```
- Analysis parameters (notebook-specific subset):
  ```python
  ZSCORE_THRESHOLD = EDA_CONFIG["zscore_threshold"]
  HISTOGRAM_BINS = EDA_CONFIG["histogram_bins"]   # or "auto" to use self-optimizing
  PERCENTILES = EDA_CONFIG["percentiles"]
  CORRELATION_HIGH_THRESHOLD = EDA_CONFIG["correlation_high_threshold"]
  ```
- Self-optimizing parameter flags:
  ```python
  AUTO_BINS = True       # True: compute bins from data; False: use HISTOGRAM_BINS
  AUTO_OUTLIER = True    # True: use adaptive IQR threshold; False: use ZSCORE_THRESHOLD
  AUTO_ACF_DEPTH = True  # True: compute max lag from data; False: use MAX_ACF_LAG
  MAX_ACF_LAG = 240      # fallback when AUTO_ACF_DEPTH is False
  ```
- Apply global matplotlib/seaborn settings:
  ```python
  sns.set_theme(style=SEABORN_STYLE)
  plt.rcParams["figure.figsize"] = FIGURE_SIZE
  ```

Document this template in a markdown cell at the top of each notebook explaining:
- What each parameter controls.
- How to switch between fixed and self-optimizing modes.
- How to change resolution selection.

Acceptance criteria:
- The configuration cell template is defined and documented.
- The template is consistent across all three notebooks (same variable names, same structure).
- All `plt.rcParams` and `sns.set_theme` calls appear only in the configuration cell, not
  scattered through analysis cells.

---

### Step 6. Define the standard notebook glossary template

Rationale: The notebooks are presentation artifacts -- they will be walked through with
the advisor and potentially included in or referenced by the final report. A reader
encountering a term like "warm-up period", "IQR", "day_class", or "rolling slope" should
not need to leave the notebook to understand it. An in-notebook glossary at the end of each
notebook makes each notebook self-contained and supports both self-guided reading and
live walkthroughs.

Tasks:

Define a standard glossary pattern for the final cell(s) of every notebook:

Final markdown cell -- Glossary:
- Title: `## Glossary`
- Format: a definition-list style table with three columns:

  ```markdown
  ## Glossary

  | Term | Definition | Context |
  |------|-----------|---------|
  | avg_load | Mean power consumption ... | Silver column, modeling target |
  | day_class | Customer-provided ... | Raw/bronze/silver/gold metadata |
  | ... | ... | ... |
  ```

- The `Term` column uses the exact variable name or concept as it appears in the notebook.
- The `Definition` column provides a 1-2 sentence plain-language explanation.
- The `Context` column notes where in the pipeline or analysis the term is relevant,
  helping readers connect the glossary entry to specific cells above.

Glossary content guidelines:
- Include every domain term that appears in the notebook (load, day_class, workday, etc.).
- Include every statistical method used (z-score, IQR, autocorrelation, ADF, VIF, etc.).
- Include every feature engineering concept referenced (lag, rolling mean, slope, warm-up
  period, delta, etc.).
- Include every pipeline concept referenced (bronze, silver, gold, resolution, resampling,
  etc.).
- Include abbreviations and their expansions (ACF, PACF, PSD, STL, MAE, RMSE, etc.).
- Order alphabetically for easy lookup.
- Keep definitions accessible -- assume the reader has basic statistics knowledge but is
  not familiar with this specific project.

Cross-reference the canonical [glossary.md](../004_reference/glossary.md) but do not
simply copy it. Each notebook glossary should be tailored to the terms actually used in
that notebook, with definitions tuned to that notebook's context.

Acceptance criteria:
- Every notebook ends with a `## Glossary` markdown cell.
- Every domain term, statistical method, and pipeline concept used in the notebook appears
  in its glossary.
- Glossary entries use plain language accessible to someone with basic statistics background.
- Glossary entries are alphabetically ordered.
- Each glossary has at least 15 entries (raw), 20 entries (bronze), 25 entries (silver),
  reflecting increasing complexity through the pipeline.

---

### Step 7. Refactor `000_raw_eda.ipynb`

Rationale: The raw EDA notebook has 7 hardcoded parameters (see initial state snapshot).
These must be moved to the configuration cell or replaced with self-optimizing calls.

Tasks:

Insert configuration cell (Cell 2, after imports):
- Declare all visualization sizes used in this notebook:
  `FIGURE_SIZE`, `FIGURE_SIZE_WIDE` (for heatmap), `FIGURE_SIZE` (for profile overlay).
- Declare analysis parameters:
  `HISTOGRAM_BINS`, `ZSCORE_THRESHOLD`, `AUTO_BINS`, `AUTO_OUTLIER`, `PERCENTILES`.
- Apply `sns.set_theme(style=SEABORN_STYLE)` and `plt.rcParams["figure.figsize"] = FIGURE_SIZE`.
- This notebook does not use resolution selection (raw data is pre-resolution).

Refactor existing cells to use configuration variables:

Daily summary statistics cell (currently Cell 5):
- No changes needed (no hardcoded parameters).

NaN heatmap cell (currently Cell 7):
- Replace `plt.figure(figsize=(14, 8))` with `plt.figure(figsize=FIGURE_SIZE_WIDE)`.

Distribution and outlier cell (currently Cell 9):
- Replace `bins=120` with:
  ```python
  bins = optimal_bin_count(flat_non_nan) if AUTO_BINS else HISTOGRAM_BINS
  ```
- Replace `> 3` sigma threshold with:
  ```python
  if AUTO_OUTLIER:
      bounds = adaptive_outlier_threshold(flat_non_nan, method="iqr")
      outlier_mask = (flat_non_nan < bounds["lower"]) | (flat_non_nan > bounds["upper"])
  else:
      outlier_mask = np.abs((flat_non_nan - mu) / sigma) > ZSCORE_THRESHOLD
  ```
- Report the threshold method and values in the print output.

Per-day load profiles cell (currently Cell 11):
- Replace `plt.figure(figsize=(14, 6))` with `plt.figure(figsize=FIGURE_SIZE)`.
- Replace hardcoded `if i < 3` label count with `if i < LEGEND_MAX_LABELS` where
  `LEGEND_MAX_LABELS = EDA_CONFIG["legend_max_labels"]` is declared in the config cell.

Remove hardcoded `plt.rcParams['figure.figsize'] = (12, 5)` from the import cell.

Add glossary cell at the end of the notebook (per Step 6 template). Include at minimum:
- `day_class`, `day_data`, `full / half / none`, `IQR`, `load`, `MATLAB (.mat)`,
  `NaN`, `NaN rate`, `outlier`, `P_data`, `percentile`, `raw layer`, `second-level data`,
  `standard deviation`, `z-score`.

Acceptance criteria:
- Zero numeric literals for figure sizes, bin counts, or thresholds appear below the
  configuration cell.
- Running with `AUTO_BINS = True` produces data-driven histogram bins.
- Running with `AUTO_BINS = False` uses the centrally defined `HISTOGRAM_BINS` value.
- Running with `AUTO_OUTLIER = True` uses IQR-based outlier detection.
- Running with `AUTO_OUTLIER = False` uses z-score with the centrally defined threshold.
- The notebook runs end-to-end without errors in both modes.
- Output is visually comparable to the current notebook when using default parameters.
- The notebook ends with a glossary cell containing at least 15 defined terms.

---

### Step 8. Refactor `001_bronze_eda.ipynb`

Rationale: The bronze EDA notebook has 8 hardcoded parameters. The z-score threshold (2.5)
differs from both the raw notebook (3.0) and `EDA_CONFIG` (3.0), creating an undocumented
inconsistency. The zero-run threshold (300 seconds) and percentile list are arbitrary with
no documented rationale.

Tasks:

Insert configuration cell (Cell 2, after imports):
- Declare visualization parameters:
  `FIGURE_SIZE`, `FIGURE_SIZE_WIDE`, `DAY_CLASS_COLORS`.
- Declare analysis parameters:
  `ZSCORE_THRESHOLD`, `AUTO_OUTLIER`, `PERCENTILES`, `OVERLAY_RESAMPLE_FREQ`,
  `ZERO_RUN_THRESHOLD_SECONDS`.
- Apply global plot settings.
- This notebook does not use resolution selection (bronze is always 1-second).

Refactor existing cells:

Day-class breakdown cell (currently Cell 4):
- No changes needed.

NaN heatmap cell (currently Cell 6):
- Replace `plt.figure(figsize=(14, 8))` with `plt.figure(figsize=FIGURE_SIZE_WIDE)`.

Outlier detection cell (currently Cell 8):
- Replace hardcoded `> 2.5` with:
  ```python
  if AUTO_OUTLIER:
      # Per-class adaptive thresholds
      ...use adaptive_outlier_threshold per group...
  else:
      daily_stats['is_outlier'] = daily_stats['mean_load_z'].abs() > ZSCORE_THRESHOLD
  ```
- Document why the threshold was previously 2.5 (or note that it was arbitrary and is now
  centralized).

All-days overlay cell (currently Cell 10):
- Replace `class_colors = {'full': '#2ca02c', ...}` with `class_colors = DAY_CLASS_COLORS`.
- Replace `.resample('5min')` with `.resample(OVERLAY_RESAMPLE_FREQ)`.
- Replace `plt.figure(figsize=(14, 6))` with `plt.figure(figsize=FIGURE_SIZE)`.

Summary statistics cell (currently Cell 12):
- Replace `percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]` with `percentiles=PERCENTILES`.

Zero-load periods cell (currently Cell 14):
- Replace `if length >= 300` with `if length >= ZERO_RUN_THRESHOLD_SECONDS`.

Remove hardcoded `plt.rcParams['figure.figsize'] = (12, 5)` from the import cell.

Add glossary cell at the end of the notebook (per Step 6 template). Include at minimum:
- `avg_load`, `bronze layer`, `day_class`, `full / half / none`, `IQR`, `load`,
  `monotonic`, `NaN`, `NaN rate`, `NaN run`, `outlier`, `parquet`, `percentile`,
  `resampling`, `second-level data`, `timestamp`, `workday`, `z-score`,
  `zero-load period`.

Acceptance criteria:
- Zero numeric literals for figure sizes, thresholds, resample frequencies, or percentile
  lists appear below the configuration cell.
- The z-score threshold is now consistent with `EDA_CONFIG["zscore_threshold"]` by default,
  eliminating the undocumented 2.5 vs 3.0 discrepancy.
- The day-class color palette is sourced from `EDA_CONFIG` and consistent across notebooks.
- The notebook runs end-to-end without errors.
- The notebook ends with a glossary cell containing at least 20 defined terms.

---

### Step 9. Refactor `002_silver_eda.ipynb`

Rationale: The silver EDA notebook has 10 hardcoded parameters and is the only notebook that
loads resolution-specific data. It currently loads only `1m`, `5m`, and `10m`, excluding the
`15m` default resolution. The resolution selection mechanism from Step 2 should be used here.

Tasks:

Insert configuration cell (Cell 2, after imports):
- Declare resolution selection:
  ```python
  RESOLUTION_MODE = "default"
  CUSTOM_RESOLUTIONS = ["5min", "15min"]
  RESOLUTIONS = resolve_eda_resolutions(RESOLUTION_MODE, CUSTOM_RESOLUTIONS)
  ```
- Declare visualization parameters:
  `FIGURE_SIZE`, `FIGURE_SIZE_WIDE`, `FIGURE_SIZE_COMPACT`, `FIGURE_SIZE_GRID`,
  `FIGURE_SIZE_CORRELATION`.
- Declare analysis parameters:
  `HISTOGRAM_BINS`, `AUTO_BINS`, `CORRELATION_HIGH_THRESHOLD`, `TOP_CORRELATIONS_COUNT`,
  `MAX_ACF_LAG`, `AUTO_ACF_DEPTH`, `DISTRIBUTION_FEATURES`.
- Apply global plot settings.

Replace hardcoded resolution loading (currently Cell 2):
- Remove the three hardcoded `pd.read_parquet(...)` calls for `1m`, `5m`, `10m`.
- Replace with a dynamic loading loop:
  ```python
  silver_data = {}
  for res in RESOLUTIONS:
      path = get_silver_path(res)
      if path.exists():
          df = pd.read_parquet(path)
          df["timestamp"] = pd.to_datetime(df["timestamp"])
          suffix = RESOLUTION_TO_SUFFIX[res]
          silver_data[suffix] = df
      else:
          print(f"Warning: Silver file not found for {res}, skipping: {path}")
  ```
- All downstream cells iterate over `silver_data` instead of referencing `silver_1m`,
  `silver_5m`, `silver_10m` by name.

Refactor resolution snapshot cell (currently Cell 4):
- Replace the manual three-row DataFrame with a loop over `silver_data`:
  ```python
  summary = pd.DataFrame([
      summarize_resolution(df, name) for name, df in silver_data.items()
  ])
  ```

Refactor correlation heatmap cell (currently Cell 6):
- Replace `plt.figure(figsize=(16, 12))` with `plt.figure(figsize=FIGURE_SIZE_CORRELATION)`.
- Replace hardcoded `head(20)` with `head(TOP_CORRELATIONS_COUNT)`.
- Use `CORRELATION_HIGH_THRESHOLD` for the high-correlation filter.
- Use the first available resolution's data (e.g., finest resolution in `silver_data`)
  rather than hardcoding `silver_1m`.

Refactor NaN cascade cell (currently Cell 8):
- Use the first available resolution's data rather than hardcoding `silver_1m`.

Refactor distribution cell (currently Cell 10):
- Replace `features_to_plot = ['avg_load', 'workday', ...]` with
  `features_to_plot = DISTRIBUTION_FEATURES`.
- Replace `bins=40` with:
  ```python
  bins = optimal_bin_count(df[col].dropna()) if AUTO_BINS else HISTOGRAM_BINS
  ```
- Replace `fig, axes = plt.subplots(2, 3, figsize=(16, 8))` with dynamic grid sizing:
  ```python
  n = len(features_to_plot)
  ncols = 3
  nrows = math.ceil(n / ncols)
  fig, axes = plt.subplots(nrows, ncols, figsize=FIGURE_SIZE_GRID)
  ```

Refactor autocorrelation cell (currently Cell 12):
- Replace `max_lag = 240` with:
  ```python
  if AUTO_ACF_DEPTH:
      max_lag = optimal_acf_depth(avg_series)
  else:
      max_lag = MAX_ACF_LAG
  ```
- Replace `plt.figure(figsize=(14, 4))` with `plt.figure(figsize=FIGURE_SIZE_COMPACT)`.
- Use the first available resolution's data rather than hardcoding `silver_1m`.

Refactor multi-resolution comparison cell (currently Cell 14):
- Replace the hardcoded three-profile comparison with a loop over `silver_data`:
  ```python
  for name, df in silver_data.items():
      profile = hourly_profile(df)
      plt.plot(profile.index, profile.values, label=name, linewidth=2)
  ```
- Replace `plt.figure(figsize=(12, 5))` with `plt.figure(figsize=FIGURE_SIZE)`.

Remove hardcoded `plt.rcParams['figure.figsize'] = (12, 5)` from the import cell.

Add glossary cell at the end of the notebook (per Step 6 template). Include at minimum:
- `ACF`, `autocorrelation`, `avg_load`, `collinearity`, `correlation`, `day_class`,
  `day_of_week`, `delta`, `feature engineering`, `full / half / none`, `gold layer`,
  `IQR`, `lag`, `multicollinearity`, `NaN`, `NaN cascade`, `PACF`, `Pearson correlation`,
  `resolution`, `rolling mean / std / max / min`, `season`, `silver layer`, `slope`,
  `time_of_day`, `VIF`, `warm-up period`, `workday`.

Acceptance criteria:
- Zero hardcoded resolution file paths remain in the notebook.
- Changing `RESOLUTION_MODE` to `"all"` loads all 8 resolutions (if silver files exist).
- Changing `RESOLUTION_MODE` to `"custom"` with `["5min", "15min"]` loads exactly those two.
- The default mode (`"default"`) loads `1min`, `5min`, `10min`, `15min`.
- All analysis cells iterate over `silver_data` rather than referencing named variables.
- Zero numeric literals for figure sizes, bin counts, lag depths, or correlation thresholds
  appear below the configuration cell.
- Running with `AUTO_BINS = True` and `AUTO_ACF_DEPTH = True` produces data-driven values.
- Running with both set to `False` uses centrally defined fallback values.
- The notebook runs end-to-end without errors in all three resolution modes.
- The notebook ends with a glossary cell containing at least 25 defined terms.

---

## Phase 3: Expanded Notebook Development

Goal: Add missing analyses and visualizations to each notebook so they meet the standard
expected for a time-series forecasting project. These additions go beyond configuration
refactoring -- they add new analytical content that supports modeling decisions, strengthens
the team's understanding of the data, and provides artifacts suitable for the capstone report.
Scope is limited to the three existing notebooks (raw, bronze, silver) per the scope boundary.

Exit criteria:
- Each notebook contains the full set of visualizations and analyses listed below.
- Every new analysis cell has an adjacent markdown cell explaining the method, what to look
  for, and what the results mean for downstream modeling.
- Interactive plots (Plotly) are used where drill-down capability adds value.
- Each notebook includes a data quality scorecard summarizing key metrics.

---

### Step 10. Expand `000_raw_eda.ipynb` with additional analyses

Rationale: The raw EDA currently covers basic shape inspection, NaN heatmap, one histogram,
and day profiles. It lacks standard diagnostics that would be expected in a professional
time-series EDA: normality testing, spectral analysis, stratified comparisons, and
quantitative stationarity checks.

Tasks:

Add after existing analyses (before glossary cell):

QQ plot for normality assessment:
- Add cell: generate a QQ (quantile-quantile) plot of raw load values against a normal
  distribution using `scipy.stats.probplot`.
- Add markdown cell explaining: QQ plots compare the observed distribution to a theoretical
  one. Departures from the diagonal line indicate non-normality. Heavy tails (common in
  load data) appear as upward/downward curves at the extremes. This matters because some
  models assume normally distributed residuals.

Power spectral density:
- Add cell: compute and plot the power spectral density (PSD) of the flattened raw load
  signal using `scipy.signal.welch` or `scipy.signal.periodogram`.
- Annotate the plot with expected periodicities: 24-hour cycle (daily), 12-hour cycle
  (business peak), and any sub-daily harmonics.
- Add markdown cell explaining: PSD reveals dominant frequencies in the signal. Strong peaks
  at 24h and 12h confirm the diurnal commercial load pattern. Unexpected peaks may indicate
  equipment cycling or other periodic behavior worth investigating.

Day-class stratified box plots:
- Add cell: side-by-side box plots of daily mean load grouped by `day_class`
  (`full`, `half`, `none`).
- Add markdown cell explaining: box plots show the central tendency, spread, and outliers
  for each business-day type. Clear separation between classes supports using `workday` as a
  predictor. Overlapping distributions suggest the classification has limited discriminative
  power for load forecasting.

Day-class transition matrix:
- Add cell: compute and display a transition matrix showing the probability of each
  `day_class` following each other `day_class` (e.g., P(full | previous=half)).
- Visualize as a heatmap.
- Add markdown cell explaining: transition probabilities reveal scheduling patterns. If
  certain transitions never occur (e.g., `none` never follows `full`), this constrains
  what the model might encounter during inference.

Interactive time-series drill-down:
- Add cell: create a Plotly interactive line chart of the full raw signal (downsampled to
  1-minute for performance) with hover tooltips showing date, hour, and load value.
- Color-code by `day_class`.
- Add markdown cell explaining: interactive visualization allows the analyst to zoom into
  anomalous periods identified in earlier cells and inspect them at higher resolution.

Data quality scorecard:
- Add cell: generate a summary table with pass/fail indicators:
  - Expected shape: pass if `(86400, 31)`.
  - NaN rate: pass if < 2% overall.
  - Date uniqueness: pass if 31 unique dates.
  - Day-class validity: pass if all values in `{full, half, none}`.
  - Physical range: pass if all non-NaN values in `[0, physical_load_max_watts]`.
  - Outlier rate: pass if < 1% of values flagged as outliers.
- Add markdown cell explaining: the scorecard provides a quick go/no-go assessment before
  proceeding to bronze ingestion. A failing check does not block processing but should be
  investigated.

Acceptance criteria:
- QQ plot, PSD, stratified box plots, transition matrix, interactive chart, and scorecard
  are all present and produce output.
- Every new cell has an adjacent markdown cell with method explanation and interpretation
  guidance.
- The notebook runs end-to-end without errors.

---

### Step 11. Expand `001_bronze_eda.ipynb` with additional analyses

Rationale: The bronze EDA currently covers day-class breakdown, NaN heatmap, NaN run
analysis, outlier detection, day-class overlay, summary statistics, and zero-load detection.
It lacks temporal decomposition, ramp-rate analysis, cross-day similarity, and interactive
exploration.

Tasks:

Add after existing analyses (before glossary cell):

Seasonal/STL decomposition:
- Add cell: compute an STL (Seasonal and Trend decomposition using Loess) decomposition of
  the daily mean load time series (31 data points) using `statsmodels.tsa.seasonal.STL`
  with `period=7` (weekly cycle).
- Plot the observed, trend, seasonal, and residual components.
- Add markdown cell explaining: STL separates the signal into trend (long-term direction),
  seasonal (repeating weekly pattern), and residual (unexplained variation). A strong seasonal
  component supports using day-of-week features. A trend component suggests the model may need
  to account for non-stationarity.

Day-class transition visualization:
- Add cell: compute and plot a transition diagram or heatmap showing which `day_class`
  follows which on consecutive days.
- Add markdown cell explaining: transition patterns reveal operational scheduling logic that
  may be predictable. For example, if weekends are always `none` and are always followed by
  `half` on Monday, the model can exploit this regularity.

Load ramp-rate distribution:
- Add cell: compute the first difference of the 1-second load signal (`load[t] - load[t-1]`)
  per day, then plot the distribution of ramp rates.
- Report the 1st, 5th, 95th, and 99th percentile ramp rates.
- Add markdown cell explaining: ramp rate measures how quickly load changes. High ramp rates
  correspond to sudden transitions (equipment start/stop) that are difficult for smooth
  forecasting models to capture. This informs whether the model needs features that capture
  transition dynamics (e.g., short-lag deltas).

Interactive per-day overlay:
- Add cell: create a Plotly interactive overlay of all 31 days (resampled to
  `OVERLAY_RESAMPLE_FREQ`) with hover tooltips showing date, time, load, and day_class.
- Add day_class as a color dimension.
- Add markdown cell explaining: interactive overlay allows the analyst to identify individual
  anomalous days and inspect load pattern deviations with precision.

Cross-day correlation matrix:
- Add cell: compute pairwise Pearson correlation between daily load profiles (each day as a
  vector of 1440 minute-level means). Display as a heatmap with day_class annotations.
- Add markdown cell explaining: high within-class correlation confirms that days of the same
  type share similar load shapes. Low correlation between a day and its class peers flags it
  as an anomaly worth inspecting.

Data quality scorecard:
- Add cell: generate a summary table with pass/fail indicators:
  - Row count: pass if `86400 * 31 = 2,678,400`.
  - Timestamp monotonicity: pass if strictly increasing.
  - NaN rate: pass if < 2%.
  - Day-class validity: pass if all values in `{full, half, none}`.
  - Duplicate timestamps: pass if zero duplicates.
  - Zero-load sustained runs: pass if no runs exceed 1 hour.
  - Physical range: pass if all non-NaN values in `[0, physical_load_max_watts]`.
- Add markdown cell explaining the scorecard.

Acceptance criteria:
- STL decomposition, transition visualization, ramp-rate distribution, interactive overlay,
  cross-day correlation, and scorecard are all present and produce output.
- Every new cell has an adjacent markdown cell with method explanation and interpretation
  guidance.
- The notebook runs end-to-end without errors.

---

### Step 12. Expand `002_silver_eda.ipynb` with additional analyses

Rationale: The silver EDA currently covers resolution snapshot, correlation heatmap, NaN
cascade, feature distributions, autocorrelation, and multi-resolution comparison. It lacks
feature importance preview, VIF analysis, partial autocorrelation, cross-resolution fidelity
measurement, stratified target analysis, and interactive exploration.

Tasks:

Add after existing analyses (before glossary cell):

Feature importance preview (mutual information):
- Add cell: compute mutual information between each numeric feature and `avg_load` using
  `sklearn.feature_selection.mutual_info_regression`. Rank and display as a horizontal bar
  chart.
- Add cell: compute absolute Pearson correlation between each feature and `avg_load`. Rank
  and display as a horizontal bar chart alongside the mutual information ranking.
- Add markdown cell explaining: mutual information captures non-linear relationships that
  Pearson correlation misses. Features ranking highly on both metrics are strong candidates
  for inclusion. Features ranking highly on MI but low on Pearson suggest non-linear effects
  that may benefit from tree-based models.

VIF (Variance Inflation Factor) analysis:
- Add cell: compute VIF for all features in the `curated` feature set using
  `statsmodels.stats.outliers_influence.variance_inflation_factor`.
- Display results as a table sorted by VIF descending. Flag features with VIF > 5 (moderate
  collinearity) and VIF > 10 (severe collinearity).
- Add markdown cell explaining: VIF quantifies how much a feature's variance is inflated by
  correlation with other features. High VIF indicates that the feature provides redundant
  information. For linear models, VIF > 10 can destabilize coefficient estimates. For
  tree-based models, multicollinearity is less harmful but can still affect feature
  importance interpretability.

Partial autocorrelation (PACF):
- Add cell: compute and plot the partial autocorrelation function (PACF) of `avg_load` at
  the finest available resolution using `statsmodels.tsa.stattools.pacf`.
- Plot alongside the existing ACF for comparison.
- Add markdown cell explaining: while ACF shows total correlation at each lag, PACF shows
  the correlation after removing the effect of intermediate lags. PACF helps determine the
  order of autoregressive models: significant PACF values at lags 1, 5, and 60 suggest these
  are the most informative direct lags, which validates the lag feature selection in the
  `curated` feature set.

Cross-resolution signal fidelity:
- Add cell: for each pair of adjacent resolutions (e.g., 1m vs 5m, 5m vs 10m), compute:
  - Mean absolute difference in hourly mean load.
  - Pearson correlation of hourly mean load profiles.
  - Percentage of variance retained (R-squared of coarser resolution predicting finer).
- Display as a table.
- Add markdown cell explaining: this quantifies information loss at each resolution step.
  If 5m retains >99% of variance compared to 1m, the 5x row reduction is a favorable
  trade-off (directly relevant to hypothesis H3).

Stratified target analysis:
- Add cell: create box plots or violin plots of `avg_load` stratified by:
  - `day_class` (full / half / none).
  - `hour` (0-23).
  - `season` (if data spans multiple seasons; if not, note this limitation).
  - `time_of_day` (morning / afternoon / evening / night).
- Add markdown cell explaining: stratified target analysis reveals which categorical features
  create meaningful load separation. Strong separation by `day_class` and `hour` supports
  hypothesis H1 and the inclusion of these features in minimal and temporal feature sets.

Per-resolution rolling statistics comparison:
- Add cell: for each resolution, compute and overlay the rolling mean and rolling standard
  deviation of `avg_load` with window=60 periods. Show one subplot per resolution.
- Add markdown cell explaining: rolling statistics at different resolutions reveal how
  smoothing changes with temporal granularity. At coarser resolutions, rolling windows cover
  longer absolute time spans, which may over-smooth fast transients.

Interactive correlation explorer:
- Add cell: create a Plotly heatmap of the feature correlation matrix with hover tooltips
  showing exact correlation values and feature names. If multiple resolutions are loaded,
  add a dropdown or tabs to switch between resolutions.
- Add markdown cell explaining: interactive exploration allows quick identification of
  highly correlated feature pairs that may need to be addressed before linear modeling.

Data quality scorecard:
- Add cell: generate a summary table with pass/fail indicators per resolution:
  - Column count: pass if 44.
  - Row count: pass if within expected range for resolution.
  - `avg_load` NaN rate: pass if < 1%.
  - Core column completeness: pass if zero NaN in non-lag columns.
  - Warm-up NaN consistency: pass if lag/rolling NaN counts match expected warm-up periods.
  - Feature correlation ceiling: pass if no pair exceeds 0.99 (near-perfect collinearity).
- Add markdown cell explaining the scorecard.

Acceptance criteria:
- Feature importance, VIF, PACF, cross-resolution fidelity, stratified target analysis,
  rolling comparison, interactive explorer, and scorecard are all present and produce output.
- Every new cell has an adjacent markdown cell with method explanation and interpretation
  guidance.
- The notebook runs end-to-end without errors in all three resolution modes.
- VIF analysis uses the `curated` feature set imported from config.
- Feature importance ranking uses features from the `full` feature set imported from config.

---

## Phase 4: TOML Configuration Migration

Goal: Migrate declarative configuration from `scripts/config.py` (Python) to TOML format,
improving readability, tooling compatibility, and separation of concerns. Computed
configuration (validation functions, column builders, self-optimizing algorithms) remains
in Python.

Exit criteria:
- All declarative configuration lives in TOML file(s).
- `scripts/config.py` reads from TOML at import time and exports the same public API.
- All pipeline scripts, notebooks, and tests continue to work without changes to their
  import statements.
- The TOML files are human-readable and editable without Python knowledge.

---

### Why TOML

The project's configuration is currently defined in `scripts/config.py` as Python data
structures (`dict`, `tuple`, `list`, `TypedDict`). While functional, this approach has
several limitations that TOML addresses:

**1. Separation of concerns.**

Configuration values (paths, thresholds, feature lists, schemas) are declarative data.
They describe *what* is configured, not *how* it is used. Embedding them in Python source
files mixes data with logic, making it harder to review settings at a glance.

TOML is a minimal, human-readable format designed specifically for configuration. It
separates the "what" (TOML) from the "how" (Python), making each easier to reason about
independently.

**2. Readability for non-developers.**

The team includes members and stakeholders who may not be fluent in Python. A TOML file
uses plain key-value syntax that anyone can read and edit:

```toml
[eda]
zscore_threshold = 3.0
histogram_bins = 50
figure_size = [14, 6]
percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
```

Compare with the current Python:
```python
EDA_CONFIG: Final[EDAConfig] = {
    "zscore_threshold": 3.0,
    "histogram_bins": 50,
    "figure_size": (14, 6),
    "percentiles": [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
}
```

The TOML version has less syntactic noise and no type annotations, import machinery, or
class definitions to distract from the actual values.

**3. Standard Python ecosystem format.**

TOML is the official format for Python project metadata (PEP 518, `pyproject.toml`). Python
3.11+ includes `tomllib` in the standard library -- no external dependencies are required
to parse it. This project requires Python 3.11+, so `tomllib` is guaranteed available.

Using TOML aligns the project with modern Python conventions and tooling expectations.

**4. Safe loading.**

Loading a Python file executes arbitrary code. Loading a TOML file parses data only. This
eliminates the (theoretical) risk of configuration files having side effects and makes it
safe to load configuration in sandboxed or restricted environments.

**5. Clean version control diffs.**

When configuration values change, TOML diffs show only the changed values. Python diffs
may include formatting changes, import reordering, or type annotation updates that obscure
the actual configuration change. TOML's flat structure produces minimal, focused diffs.

**6. Tooling compatibility.**

TOML files can be validated by schema tools, linted by dedicated formatters (`taplo`),
and consumed by non-Python tools (CI/CD systems, documentation generators, monitoring
dashboards). Python configuration files require Python-specific tooling for any automated
processing.

**7. Consolidation opportunity.**

The project currently has no `pyproject.toml`. Migrating to TOML creates an opportunity
to consolidate project metadata (name, version, dependencies, tool configuration) alongside
pipeline configuration in a standard location. This reduces the number of configuration
files a new contributor must understand.

---

### Step 13. Design TOML configuration structure

Rationale: Before migrating, the team must agree on the TOML file layout: whether to use a
single `pyproject.toml` with custom sections, a standalone `config/pipeline.toml`, or
multiple per-concern files. The choice affects discoverability, merge conflicts, and tooling.

Tasks:

Evaluate and recommend a file layout. Recommended structure:

```text
electric-load-forecasting/
|-- pyproject.toml              Project metadata, tool config (pytest, etc.)
|-- config/
|   |-- pipeline.toml           Pipeline paths, resolutions, schemas, feature config
|   `-- eda.toml                EDA visualization and analysis defaults
```

Rationale for this layout:
- `pyproject.toml` handles standard Python project metadata and tool configuration
  (`[project]`, `[tool.pytest]`, etc.). This is the conventional location.
- `config/pipeline.toml` contains pipeline-specific declarative configuration (paths,
  resolutions, feature windows, schemas, day-class maps, split ranges, feature sets).
  Separating from `pyproject.toml` avoids overloading a single file and keeps pipeline
  config independent of Python packaging concerns.
- `config/eda.toml` contains EDA-specific visualization and analysis defaults. Separating
  from pipeline config makes it clear that EDA settings are presentation concerns, not
  data contracts.

Define the TOML schema for `config/pipeline.toml`:

```toml
[paths]
raw_mat = "data/000_raw/P_data.mat"
bronze_file = "data/001_bronze/power_load_1s.parquet"
silver_dir = "data/002_silver"
gold_dir = "data/003_gold"
model_dir = "data/004_model"
logs_dir = "logs"

[resolutions]
supported = ["1s", "5s", "10s", "30s", "1min", "5min", "10min", "15min"]
defaults = ["1min", "5min", "10min", "15min"]

[resolutions.aliases]
"60s" = "1min"

[resolutions.suffixes]
"1s" = "1s"
"5s" = "5s"
"10s" = "10s"
"30s" = "30s"
"1min" = "1m"
"5min" = "5m"
"10min" = "10m"
"15min" = "15m"

[features]
lag_periods = [1, 5, 15, 60, 1440]
rolling_periods = [5, 15, 60, 240, 1440]
slope_periods = [5, 15, 60]

[day_class]
mapping = { none = 0, half = 1, full = 2 }
valid_classes = ["full", "half", "none"]

[splits]
train = [1, 25]
validate = [26, 28]
test = [29, 31]

[target]
column = "avg_load"

[feature_sets.minimal]
columns = ["workday", "hour", "lag_1"]

[feature_sets.temporal]
columns = [
    "workday", "year", "quarter", "month", "day",
    "day_of_week", "hour", "season", "time_of_day", "lag_1"
]

[feature_sets.curated]
columns = [
    "workday", "hour", "season", "time_of_day",
    "lag_1", "lag_5", "lag_60", "lag_1440",
    "rolling_mean_15", "rolling_std_60", "slope_15"
]

# "full" feature set is computed (all columns minus metadata and target),
# so it remains in Python rather than TOML.
```

Define the TOML schema for `config/eda.toml`:

```toml
[visualization]
figure_size = [14, 6]
figure_size_wide = [14, 8]
figure_size_compact = [14, 4]
figure_size_grid = [16, 8]
figure_size_correlation = [16, 12]
seaborn_style = "whitegrid"

[visualization.day_class_colors]
full = "#2ca02c"
half = "#1f77b4"
none = "#ff7f0e"

[analysis]
zscore_threshold = 3.0
histogram_bins = 50
correlation_high_threshold = 0.95
top_correlations_count = 20
zero_run_threshold_seconds = 300
overlay_resample_frequency = "5min"
legend_max_labels = 3
percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

[analysis.physical_range]
load_min_watts = 0.0
load_max_watts = 100000.0

[analysis.distribution_features]
columns = ["avg_load", "workday", "day_of_week", "hour", "season", "time_of_day"]

[resolution_selection]
default_mode = "default"
```

Acceptance criteria:
- TOML schema is documented and reviewed by the team.
- Every key in the current `scripts/config.py` has a corresponding TOML location.
- Computed values (`SILVER_COLUMNS`, `full` feature set, validation functions) are
  identified as "remains in Python".

---

### Step 14. Implement TOML loading in `scripts/config.py`

Rationale: The migration must be transparent to all consumers. `scripts/config.py` continues
to be the import target for all scripts, notebooks, and tests. Internally, it reads from
TOML files instead of defining values inline.

Tasks:

Create `config/pipeline.toml` and `config/eda.toml` with the schemas from Step 13.

Update `scripts/config.py`:
- Add TOML loading at the top of the module:
  ```python
  import tomllib
  _CONFIG_DIR = PROJECT_ROOT / "config"
  with open(_CONFIG_DIR / "pipeline.toml", "rb") as f:
      _PIPELINE_TOML = tomllib.load(f)
  with open(_CONFIG_DIR / "eda.toml", "rb") as f:
      _EDA_TOML = tomllib.load(f)
  ```
- Replace inline `PATHS` dictionary with values read from `_PIPELINE_TOML["paths"]`.
  Convert relative path strings to absolute `Path` objects resolved against `PROJECT_ROOT`.
- Replace inline `SUPPORTED_RESOLUTIONS` with `tuple(_PIPELINE_TOML["resolutions"]["supported"])`.
- Replace inline `DEFAULT_RESOLUTIONS` with `tuple(_PIPELINE_TOML["resolutions"]["defaults"])`.
- Replace inline `RESOLUTION_ALIASES` with `dict(_PIPELINE_TOML["resolutions"]["aliases"])`.
- Replace inline `RESOLUTION_TO_SUFFIX` with `dict(_PIPELINE_TOML["resolutions"]["suffixes"])`.
- Replace inline `FEATURE_CONFIG` with values from `_PIPELINE_TOML["features"]`.
- Replace inline `DAY_CLASS_MAP` with values from `_PIPELINE_TOML["day_class"]["mapping"]`.
- Replace inline `SPLIT_DAY_RANGES` with values from `_PIPELINE_TOML["splits"]`, converting
  lists to tuples.
- Replace inline `TARGET_COLUMN` with `_PIPELINE_TOML["target"]["column"]`.
- Replace inline `FEATURE_SETS` for `minimal`, `temporal`, and `curated` with values from
  `_PIPELINE_TOML["feature_sets"]`. The `full` set remains computed in Python.
- Replace inline `EDA_CONFIG` with values from `_EDA_TOML`, restructured to match the
  current dictionary shape.

Preserve the existing public API:
- All existing module-level names (`PATHS`, `SUPPORTED_RESOLUTIONS`, `EDA_CONFIG`,
  `FEATURE_SETS`, `SCHEMAS`, `SILVER_COLUMNS`, `validate_config`, etc.) must continue
  to exist with the same types and values.
- No consumer code (scripts, notebooks, tests) should require import changes.

Keep computed values in Python:
- `_build_silver_columns()` remains in Python (derives column list from feature config).
- `SILVER_COLUMNS` remains computed.
- `SCHEMAS` remains constructed in Python from TOML-sourced values.
- `full` feature set remains computed (all columns minus metadata and target).
- `validate_config()` remains in Python.
- Self-optimizing functions remain in Python.

Add TOML file existence validation:
- If either TOML file is missing, raise `FileNotFoundError` with the expected path and a
  message suggesting the user may need to run from the project root.

Acceptance criteria:
- `config/pipeline.toml` and `config/eda.toml` exist and contain all declarative config.
- `scripts/config.py` reads from TOML and exports the same public API.
- All existing tests pass without modification.
- All pipeline scripts run without modification.
- All notebooks run without modification.
- `validate_config()` continues to catch the same errors.
- Removing a key from TOML produces a clear `KeyError` traceable to the TOML file.

---

### Step 15. Create `pyproject.toml` for project metadata

Rationale: The project has no `pyproject.toml`. Creating one consolidates project metadata
and tool configuration in the standard Python location. This is a natural companion to the
TOML migration and improves the project's conformance to modern Python conventions.

Tasks:

Create `pyproject.toml` at the project root:

```toml
[project]
name = "electric-load-forecasting"
version = "0.1.0"
description = "Short-term electric load forecasting pipeline and modeling framework"
requires-python = ">=3.11"

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.coverage.run]
source = ["scripts", "run_pipeline"]

[tool.coverage.report]
show_missing = true
```

Declare runtime and development dependencies directly in `pyproject.toml` so there is
a single dependency source of truth.

Acceptance criteria:
- `pyproject.toml` exists at the project root.
- `pytest` discovers and runs tests via `pyproject.toml` configuration.
- Project metadata is accurate.

---

### Step 16. Add tests for TOML configuration loading

Rationale: The TOML migration introduces a new loading path that must be validated to ensure
no configuration values were lost or changed during migration.

Tasks:

Add to `tests/unit/test_config.py`:
- Test that `config/pipeline.toml` exists and is valid TOML.
- Test that `config/eda.toml` exists and is valid TOML.
- Test that every key in the TOML files maps to an expected module-level export in
  `scripts/config.py`.
- Test that `PATHS` values are `Path` objects (not strings).
- Test that `SPLIT_DAY_RANGES` values are tuples (not lists).
- Test round-trip consistency: values in TOML match the exported Python constants.
- Test that modifying a TOML value (via a temporary copy) changes the loaded config
  (verifies TOML is actually being read, not bypassed).
- Test that a missing TOML file produces `FileNotFoundError` with an informative message.

Acceptance criteria:
- All new tests pass.
- No existing tests are broken.
- TOML loading path is covered by at least 5 dedicated tests.

---

## Phase 5: Validation and Documentation

Goal: Verify that all changes (configuration, notebook expansion, glossaries, TOML migration)
work together correctly, and update all cross-references.

Exit criteria:
- All notebooks pass `validate_notebooks.py` end-to-end.
- All tests pass.
- Documentation is updated.
- Changelog records all changes.

---

### Step 17. Full notebook execution validation

Rationale: Phases 2-3 make extensive changes to all three notebooks. Every notebook must
be re-executed end-to-end to verify correctness.

Tasks:

- Run `python scripts/validate_notebooks.py` to execute all three notebooks end-to-end.
- For the silver notebook, test all three resolution modes.
- Verify that `AUTO_*` flags work in both `True` and `False` modes.
- Verify that all new analyses (Phase 3) produce output without errors.
- Verify that all glossary cells render correctly.
- Verify that interactive Plotly charts render in the notebook output.

Acceptance criteria:
- `python scripts/validate_notebooks.py` exits with code 0.
- No notebook produces warnings about missing variables or undefined names.
- Self-optimizing parameters produce values within documented bounds.
- Plotly charts produce visible output (HTML/JSON in cell output).

---

### Step 18. Update documentation and changelog

Rationale: The configuration architecture, TOML migration, notebook development expansion,
and notebook glossaries must be documented so teammates understand the changes.

Tasks:

Update [architecture.md](../001_architecture/000_overview/architecture.md):
- Add a subsection under "Shared Infrastructure" documenting `EDA_CONFIG`, the resolution
  selection utilities, and the TOML configuration files.
- Document the three resolution modes (`all`, `default`, `custom`).
- Document the TOML file locations and their relationship to `scripts/config.py`.

Update [pipeline.md](../002_pipeline/pipeline.md):
- Add a section documenting notebook configuration: where parameters are defined, how to
  override them, and how resolution selection works.
- Add a section documenting the TOML migration and file layout.

Update [glossary.md](../004_reference/glossary.md):
- Add entries for: `self-optimizing parameter`, `resolution mode`, `configuration cell`,
  `EDA_CONFIG`, `TOML`, `pyproject.toml`, `Freedman-Diaconis rule`, `IQR`,
  `mutual information`, `VIF`, `STL decomposition`, `power spectral density`,
  `partial autocorrelation`, `data quality scorecard`.

Update the spec-specific changelog at
[docs/change logs/001spec/changelog.md](../change%20logs/001spec/changelog.md):
- Add entry for each phase completed, with a summary of all changes.
- The root [changelog.md](../../changelog.md) remains an index only -- do not add
  implementation details there.

Update `README.md`:
- Add `config/` directory to the project structure tree.
- Note the TOML configuration files in the setup section.

Acceptance criteria:
- Architecture doc includes TOML and notebook configuration documentation.
- Pipeline doc includes notebook configuration and TOML sections.
- Glossary includes all new terms.
- Changelog records the changes.
- README includes `config/` directory.

---

## Parameter governance summary

The following table maps every previously hardcoded parameter to its governance strategy
under this specification:

| Parameter | Governance | Source | Self-Optimizing Alternative |
|-----------|-----------|--------|----------------------------|
| Figure sizes | Central | `config/eda.toml [visualization]` | None (aesthetic choice) |
| Seaborn style | Central | `config/eda.toml [visualization]` | None (aesthetic choice) |
| Day-class colors | Central | `config/eda.toml [visualization.day_class_colors]` | None (semantic mapping) |
| Histogram bins | Central + Self-optimizing | `config/eda.toml [analysis]` | `optimal_bin_count()` via Freedman-Diaconis |
| Z-score threshold | Central + Self-optimizing | `config/eda.toml [analysis]` | `adaptive_outlier_threshold()` via IQR/MAD |
| Percentiles | Central | `config/eda.toml [analysis]` | None (standard quantiles) |
| ACF max lag | Central + Self-optimizing | Config cell `MAX_ACF_LAG` | `optimal_acf_depth()` via significance bound |
| Correlation threshold | Central | `config/eda.toml [analysis]` | None (domain convention) |
| Top correlations count | Central | `config/eda.toml [analysis]` | None (display preference) |
| Zero-run threshold | Central | `config/eda.toml [analysis]` | None (domain-specific) |
| Overlay resample freq | Central | `config/eda.toml [analysis]` | None (display preference) |
| Legend max labels | Central | `config/eda.toml [analysis]` | None (display preference) |
| Distribution features | Central | `config/eda.toml [analysis.distribution_features]` | None (analysis scope) |
| Resolutions to load | Central + Selectable | `resolve_eda_resolutions()` | None (user choice) |
| Pipeline paths | Central | `config/pipeline.toml [paths]` | None (infrastructure) |
| Feature windows | Central | `config/pipeline.toml [features]` | None (domain design) |
| Split ranges | Central | `config/pipeline.toml [splits]` | None (domain design) |
| Schemas | Computed | Python (`scripts/config.py`) | None (derived from feature config) |

## Out of scope

- **Any notebook beyond silver.** No gold EDA, model evaluation, or cross-layer summary
  notebooks. If a need arises, ask the team first (see scope boundary above).
- **New notebooks of any kind** without explicit team approval.
- Notebook widget/interactive controls (ipywidgets) beyond Plotly. Configuration is
  code-based, not GUI-based.
- Parameter optimization via grid search or Bayesian methods (reserved for modeling phase).
- Automated notebook parameterization via Papermill or similar tools.
- Introducing duplicate dependency manifests outside `pyproject.toml`.
- Libraries without confirmed ARM64 macOS wheel availability (see platform compatibility).

## Recommended execution order

1. Phase 1 Steps 1-2 (extend config, add resolution utilities).
2. Phase 1 Step 3 (self-optimizing parameter functions).
3. Phase 1 Step 4 (tests for new infrastructure).
4. Phase 2 Steps 5-6 (define config cell and glossary templates).
5. Phase 2 Steps 7-9 (refactor each notebook: config cells + glossaries).
6. Phase 3 Steps 10-12 (expand analyses in each notebook).
7. Phase 4 Steps 13-16 (TOML migration + pyproject.toml + tests).
8. Phase 5 Steps 17-18 (validation and documentation).

## Definition of done

This specification is complete when:
- Every notebook configuration parameter is either centrally defined or self-optimized.
- Resolution selection is controllable from a single cell in every resolution-aware notebook.
- No hardcoded numeric literals for analysis parameters exist below the configuration cell
  in any notebook.
- Every notebook ends with a self-contained glossary.
- Every notebook includes the full set of analyses specified in Phase 3.
- All declarative configuration lives in TOML files under `config/`.
- `scripts/config.py` reads from TOML and exports the same public API.
- All notebooks pass `validate_notebooks.py` with default configuration.
- All tests pass.
- Self-optimizing parameter functions are tested and documented.
- `pyproject.toml` exists with project metadata and tool configuration.
- Documentation and changelog are updated.

## Document map

- Pipeline hardening spec: [000_spec.md](000_spec.md)
- Architecture overview: [architecture.md](../001_architecture/000_overview/architecture.md)
- Pipeline operations: [pipeline.md](../002_pipeline/pipeline.md)
- Feature sets: [feature_sets.md](../003_modeling/feature_sets.md)
- Glossary: [glossary.md](../004_reference/glossary.md)
- Spec changelog: [docs/change logs/001spec/changelog.md](../change%20logs/001spec/changelog.md)
- Changelog index: [changelog.md](../../changelog.md)
