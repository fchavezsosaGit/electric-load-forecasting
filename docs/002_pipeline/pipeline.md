# Data Pipeline

This document defines the repository pipeline from raw ingestion to model-ready datasets.
The pipeline supports the project goal of delivering optimizer-ready load predictions
across nowcast and forecast horizons, as tested through hypotheses H1-H5
(see [hypothesis.md](../003_modeling/hypothesis.md)).

For detailed layer-by-layer documentation, see the architecture docs listed below.

Related references:
- Execution specification: [000_spec.md](../000_governance/000_spec.md)
- Notebook configurability specification: [001_spec.md](../000_governance/001_spec.md)
- Operating direction specification: [002_operating_direction_spec.md](../000_governance/002_operating_direction_spec.md)
- Architecture overview: [architecture.md](../001_architecture/000_overview/architecture.md)
- Stage map: [stage_map.md](stage_map.md)
- Current validation snapshot: [current_validation_snapshot.md](../003_modeling/current_validation_snapshot.md)
- Feature set definitions: [feature_sets.md](../003_modeling/feature_sets.md)
- Model and blend guide: [model_and_blend_guide.md](../003_modeling/model_and_blend_guide.md)
- Current operating summary: [current_operating_approach.md](../003_modeling/current_operating_approach.md)
- Report IV run summary: [report_iv_run_summary.md](../003_modeling/report_iv_run_summary.md)
- Report IV success scorecard: [report_iv_success_scorecard.md](../003_modeling/report_iv_success_scorecard.md)
- Glossary: [glossary.md](../004_reference/glossary.md)

## Layers

| Layer | Entry Script | Canonical Implementation | Detail Doc |
|-------|--------------|--------------------------|------------|
| Raw | N/A (read-only) | N/A | [raw.md](../001_architecture/001_raw/raw.md) |
| Bronze | `scripts/000_raw_to_bronze.py` | `scripts/stages/raw_to_bronze.py` | [bronze.md](../001_architecture/002_bronze/bronze.md) |
| Silver | `scripts/001_bronze_to_silver.py` | `scripts/stages/bronze_to_silver.py` | [silver.md](../001_architecture/003_silver/silver.md) |
| Gold | `scripts/002_silver_to_gold.py` | `scripts/stages/silver_to_gold.py` | [gold.md](../001_architecture/004_gold/gold.md) |
| Model | `scripts/003_create_model_datasets.py` | `scripts/stages/create_model_datasets.py` | [model.md](../001_architecture/005_model/model.md) |

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

## Execution View

The repository now operates in four layers of evidence:
- data layers: raw -> bronze -> silver -> gold -> model datasets
- notebook and Stage-4 modeling: validate core exploratory and benchmark outputs
- horizon-specific selection: Stage-5, Stage-6, and Stage-7 keep the strongest
  measured policy for each horizon instead of forcing one model to win everywhere
- end-to-end control replay: Stage-8 summarizes the cross-horizon envelope and
  Stage-10 tests whether the current day-ahead plus intraday stack actually reduces
  locked-interval error on shared control cycles

That separation is deliberate. A good day-ahead profile model can still be a poor
last-mile correction model, and a strong short-horizon correction model does not
automatically become the right `24h` profile model.

If you only need the current validated answer, start with
[current_validation_snapshot.md](../003_modeling/current_validation_snapshot.md).
If the labels or wrappers in that snapshot feel dense, read
[model_and_blend_guide.md](../003_modeling/model_and_blend_guide.md) next.
The rest of this document explains how the pipeline produces that state.

## Plain-English Stage Map

The repo keeps stable stage ids because output folders, manifests, and report
citations depend on them. To make those ids easier to understand without
breaking provenance, use this plain-English map:

- Stage-0 to Stage-3: data preparation (`raw` -> `bronze` -> `silver` -> `gold`)
- Stage-4: notebook benchmark surface
- Stage-5: short-horizon holdout gate
- Stage-6: matched-horizon comparison
- Stage-7: recursive rollout selection and challenger sweeps
- Stage-8: horizon capability curve
- Stage-10: control-loop backtest

The fuller "what question does this stage answer?" version lives in
[stage_map.md](stage_map.md).
`outputs/008_notebook_runs/` is a notebook evidence archive, not a separate
model-selection stage.

## Measurement Surface

The pipeline measures quality with different objective families because each
stage answers a different operational question:

- Stage-4 and Stage-5: `MAE`, `RMSE`, `MAE%`, `RMSE%`, coverage, and holdout
  comparisons answer whether the short-horizon learned model is credible at all.
- Stage-6: `endpoint_mae` and `path_mae`, plus their normalized percentages,
  answer which resolution/model pair is strongest at the same representable
  horizon.
- Stage-7: `next_lock_mae`, `phase_mean_mae`, `path_mae`, and
  `profile_shape_mae` answer which rollout policy is strongest for the requested
  objective.
- Stage-8: the horizon curve summarizes the best measured candidate at each
  horizon for each objective family.
- Stage-10: forecast-control backtesting measures whether the layered update
  policy actually reduces locked-interval and full-profile error relative to a
  frozen day-ahead forecast.
- Stage-10 also now runs a day-ahead refresh study that compares:
  - frozen day-ahead
  - unconditional residual refresh
  - triggered residual refresh
- Stage-10 also records a phase stack guard summary so a `15m` winner that
  looks good in isolation is only kept when it still helps after the hourly
  layer is already applied.
- Stage-10 now also tests phase baseline-control families that blend a learned
  `15m` candidate toward the best reconstructable baseline from the same replay
  family, including optional 5-minute bucket weights inside the correction
  window.

Stage-10 replay policy:
- the control backtest benchmarks a small candidate pool using the selected
  learned policy plus baseline comparators
- once that benchmark chooses the control-layer winner, the full control replay
  materializes only the selected policy rather than the entire learned family
- the day-ahead refresh study replays a dedicated residual candidate on hourly
  checkpoints and promotes `triggered_refresh` only when it improves
  `profile_shape_mae` without worsening `lock_mae`, while still preserving
  enough of the unconditional-refresh gain to justify the extra trigger logic
- when the exact-control phase calibration surface is too thin to support a
  separate blend search, the phase baseline-control blend logic now falls back
  to the held-out phase benchmark surface instead of silently emitting no
  candidate family

Operational note:
- Stage-10 now keeps an exact-origin replay cache under
  `outputs/010_forecast_control/<artifact_namespace>/replay_cache/`, so repeated
  control backtests can reuse the same Stage-7 replay artifacts instead of
  recomputing them
- exact `15m` control replay is still the most expensive layer on a cold cache,
  but repeated Stage-10 runs are now practical in a normal developer loop

Normalization rule:
- raw MAE remains the native operational unit
- MAE% and related percentages are required whenever horizons, facilities, or
  candidate families are compared

Promotion rule:
- registry promotion should be based on the stage objective and the candidate's
  ability to beat persistence or the best baseline on that same objective, not
  on a single global leaderboard

## Visualization Surface

The pipeline produces visuals in layers so operators can move from a summary
readout to the underlying evidence:

- Stage-4 notebook figures explain fit quality, time-of-day error structure, and
  the initial model ranking.
- Stage-8 horizon-curve figures explain where learned models help or fail as
  horizon grows.
- Stage-10 control figures explain whether intraday updates reduce the next
  locked interval error and improve the overall day profile.
- Stage-10 refresh figures explain whether the learned day-ahead residual path
  is useful as a conditional correction layer, even when it is not the best
  standalone `1440m` rollout.

Each decision-facing figure bundle now includes a sibling `figure_guide.md`
written into the same output directory. The guide documents the visualization's
intent, how to read it, and what to look for when judging success or failure.

Notebook evidence is also archived under `outputs/008_notebook_runs/` before
tracked outputs are cleared, so figure interpretation can always be tied back to
the executed notebook snapshot and artifact manifest that produced it.

## Artifact Retention

Because the repo uses timestamped run directories for provenance, repeated
experimentation can leave a large number of stale outputs behind. Cleanup is now
handled by `scripts/tooling/cleanup_outputs.py` instead of ad hoc deletion.

Retention policy:

- keep `latest/` aliases and stage support folders
- keep a small recent buffer per stage root
- keep any dated run still referenced by the current docs or the current latest
  artifact surface
- prune only superseded dated runs outside that keep set

The cleanup tool writes both markdown and JSON reports under `personal/` so the
current retention decision remains auditable.

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

`scripts/validate_notebooks.py` executes core notebooks (`000`-`003`) with default
settings, refreshes silver/gold/model inputs before `003_modeling.ipynb`, and runs a
silver validation matrix covering `default`, `all`, and `custom` resolution modes plus
both automatic and fixed-parameter behaviors.

## TOML Configuration Layout

Declarative configuration is stored under `config/`:
- `config/pipeline.toml`: pipeline paths, resolution policy, legacy period-based and
  time-normalized feature windows, Fourier cycle specs, day-class mapping, split
  ranges, target, feature sets, raw ingestion contract, and stage quality thresholds.
- `config/eda.toml`: notebook visualization and analysis defaults, physical range bounds,
  and default notebook resolution mode.
- `config/modeling.toml`: shared Stage-5/Stage-6 joblib runtime controls (backend,
  worker caps, batching, dispatch policy, and per-stage toggles), adaptive HGB search
  settings, horizon policies, and segmented evaluation controls.
- `config/multires.toml`: Stage-6/Stage-10 output roots, comparison modes, horizon lists,
  baseline toggles, selection gates, rollout defaults, rollout challenger origin-policy
  diversity, horizon-curve defaults, forecast-control defaults, and mode-specific profile scopes.

`scripts/config.py` is the stable runtime API. It loads all runtime TOML files with `tomllib`,
normalizes types (for example path strings to `Path` objects and split lists to tuples),
builds computed values (`SILVER_COLUMNS`, `SCHEMAS`, `full` feature set), and enforces
runtime validation through `validate_config()`.

Script organization:
- numbered root scripts remain stable entrypoints and compatibility wrappers
- `scripts/stages/` contains medallion/data-stage implementations
- `scripts/modeling/` contains stage-5 through stage-7 implementations and helpers
- `scripts/modeling/parallel.py` centralizes the shared joblib execution plan used by
  Stage-5 and Stage-6
- `scripts/tooling/` contains environment bootstrap, notebook validation, and E2E tooling

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

Schema (89 columns):
- Core: `timestamp`, `avg_load`, `day_class`
- Business: `workday`
- Temporal: `year`, `quarter`, `month`, `day`, `day_of_week`, `hour`, `season`, `time_of_day`
- Fourier: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`
- Phase context: `phase_minute_15m`, `phase_progress_15m`, `phase_boundary_dist_15m`,
  `phase_boundary_flag_15m`, `phase_sin_15m`, `phase_cos_15m`
- Lag: `lag_1`, `lag_5`, `lag_15`, `lag_60`, `lag_1440`
- Rolling: `rolling_mean_*`, `rolling_std_*`, `rolling_max_*`, `rolling_min_*`
- Delta: `delta_5`, `delta_15`, `delta_60`, `delta_1440`
- Slope: `slope_5`, `slope_15`, `slope_60`
- Time-normalized: `lag_min_*`, `rolling_mean_min_*`, `rolling_std_min_*`,
  `rolling_max_min_*`, `rolling_min_min_*`, `slope_min_*`
- Baseline/regime: `previous_day_load`, `avg_workday_baseline`,
  `anchored_workday_baseline`,
  `profile_residual_lag_1`, `previous_day_residual`, `prev_day_workday`,
  `next_day_workday`, `workday_transition`, `profile_activity_ratio`,
  `profile_active_flag`

Notes:
- Legacy lag/rolling/slope windows remain period-based and therefore scale with resolution.
- Time-normalized window families preserve a fixed lookback in minutes across resolutions.
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
python run_pipeline.py --stage multires --multires-mode smoke
python run_pipeline.py --stage rollout_sweep
python run_pipeline.py --stage rollout
python run_pipeline.py --stage horizon_curve
python run_pipeline.py --stage forecast_control
python scripts/run_e2e.py --mode quick
python scripts/run_e2e.py --mode full
python scripts/run_e2e.py --mode quick --with-multires --with-rollout
python scripts/run_e2e.py --mode quick --with-multires --with-rollout --with-rollout-sweep
python scripts/run_e2e.py --mode quick --with-multires --with-rollout --with-rollout-sweep --with-horizon-curve --with-forecast-control
python run_pipeline.py --dry-run
```

Run all data stages and include Stage-5 performance evaluation:

```bash
python run_pipeline.py --stage all --include-performance --performance-mode quick
```

Run all data stages plus the post-MVP multiresolution stack:

```bash
python run_pipeline.py --stage all --include-multires --multires-mode smoke
python run_pipeline.py --stage all --include-performance --performance-mode quick --include-multires --include-rollout
```

Performance mode details:
- `quick`: evidence-dense short-horizon path driven by centralized quick profiles in
  `config/modeling.toml` (currently `minimal_phase_anchor`, `full_stable`,
  `curated_ramp`, `minimal_phase` plus frontier HGB variants for `1m`).
- `full`: full fold grid with horizon-policy feature/model selection, adaptive HGB
  screening, segmented holdout evaluation, and blend guardrail output.
- `preflight`: protocol checks only, no fold training.

Root wrappers are available for both shells:
- `./run_e2e.sh`
- `.\run_e2e.ps1`
- `PYTHON_BIN=python3.12 ./run_e2e.sh --mode quick`
- `.\run_e2e.ps1 -PythonExe py -- --mode quick`

Notebook smoke validation:

```bash
python scripts/validate_notebooks.py
python scripts/write_validation_snapshot.py
```

Validation behavior notes:
- Notebook execution uses a Python-managed nbconvert runner in
  `scripts/validate_notebooks.py` so Windows runs apply selector event-loop policy
  automatically and avoid prior `zmq` runtime warnings.
- Notebook validation now clears transient cell outputs after successful execution by
  default so tracked notebooks do not retain machine-specific warning paths or local
  runtime noise. Use `--keep-output` only when notebook output retention is intentional.
- Before tracked notebook outputs are cleared, the executed notebook snapshot is archived
  under `outputs/008_notebook_runs/<run_id>/`; `outputs/008_notebook_runs/latest/`
  mirrors the latest successful archive run.
- The validator refreshes silver, gold, and model datasets before
  `notebooks/003_modeling.ipynb`, so modeling notebook validation does not depend on
  stale intermediate artifacts.

## Centralized Optimization Surfaces

The pipeline now centralizes the main self-optimizing decisions instead of leaving them
as ad hoc one-off tuning choices:

- Stage-5 horizon policies map forecast horizon buckets to feature-set eligibility,
  model-family eligibility, and residual/blend rules.
- Adaptive HGB search screens candidate tree settings before the full fold run and
  records the evidence in `adaptive_hgb_screen.csv`.
- Stage-5 holdout evaluation records `holdout_segment_evaluation.csv` so model quality
  can be checked by regime, not only by aggregate MAE.
- Stage-6 winner selection records both raw MAE and normalized MAE% in
  `selection_summary.csv` and `winner_registry.csv`.
- Stage-7 rollout selection is objective-aware (`path_mae` vs `endpoint_mae`) and now
  promotes winners from measured artifact registries rather than from mutable config alone.
- Stage-7 challenger sweeps can span multiple rollout-origin policies in one pass
  (`uniform`, `billing_aligned`, `phase_balanced`, `midnight`) so short- and
  long-horizon selection is less sensitive to one origin rule.
- Default smoke scope includes `000_raw_eda.ipynb`, `001_bronze_eda.ipynb`,
  `002_silver_eda.ipynb`, and `003_modeling.ipynb`.
- Silver notebook validation includes baseline plus three profile runs:
  `default`, `all`, and `custom`.
- Stage-5 performance is executed through `scripts/004_model_performance.py`
  (canonical implementation: `scripts/modeling/model_performance.py`) and summarized back into
  `003_modeling.ipynb` when artifacts exist.
- Stage-4 modeling outputs are scoped under `outputs/004_modeling/<artifact_namespace>/`.
  These CSVs now include `mae_pct` / `rmse_pct`, defined as
  `100 * error / mean(abs(actual_load))` over valid rows.
- Stage-5 writes timestamped artifacts under `outputs/005_performance/<artifact_namespace>/` and refreshes
  `outputs/005_performance/<artifact_namespace>/latest/` as the convenience alias for the newest successful run.
- Stage-5 fold evaluation uses the shared runtime defined in `config/modeling.toml`;
  the resolved worker plan is written into `outputs/005_performance/<artifact_namespace>/latest/run_manifest.json`.
- Stage-5 now also writes:
  - `coverage_audit.csv` for split-level feature-set coverage
  - `promotion_candidate.json` for the exact candidate promoted from the fold scoreboard
- `blend_finalists.csv` for the best validation-selected guarded blend config in each shortlisted Stage-5 learned family
  - `holdout_evaluation.csv` for promoted-candidate vs baseline holdout comparison
  - `holdout_predictions.csv` for the exact one-shot holdout prediction path
  - `holdout_inference.csv` plus `fig_holdout_benchmark_ci.png` for moving-block bootstrap confidence intervals and paired significance tests
  - `deployment_recommendation.json` for the current operational winner decision
  - `feature_importance_permutation.csv`, `feature_importance_summary.json`, and `fig_feature_importance.png` for learned-challenger interpretation
  - `holdout_registry.csv` for cross-run Stage-5 learned holdout winners and their
    saved blend settings
- `full_stable` is now a shared canonical feature set derived from `full` by removing
  the unstable `rolling_*_240` and `rolling_*_1440` windows. Current evidence shows
  that this is the first Stage-5 path that beats persistence on the one-shot holdout
  split, and it is now available to Stage-3/Stage-6/Stage-7 rather than only Stage-5.
- When Stage-5 is invoked through `run_pipeline.py` and step-4 artifacts are
  missing, the orchestrator now bootstraps model dataset generation plus the
  `003_modeling.ipynb` artifact export before running performance.
- Stage-6 multiresolution comparison is executed through
  `scripts/005_multires_compare.py` (canonical implementation:
  `scripts/modeling/multires_compare.py`) or `run_pipeline.py --stage multires`, and
  writes timestamped artifacts under `outputs/006_multires/<artifact_namespace>/` plus a `latest/`
  alias.
- Stage-6 native-step and matched-horizon task grids use the same shared runtime from
  `config/modeling.toml`; the resolved plans are recorded in the multires run manifest.
- Stage-6 now skips missing configured gold inputs with explicit
  `skipped_missing_resolution:*` warnings rather than failing mid-run when the remaining
  requested resolutions are still runnable.
- Stage-6 deduplicates raw `fold_metrics.csv` and `origin_metrics.csv` before writing so
  baseline rows keep one-row-per-candidate semantics.
- Stage-6 writes `winner_registry.csv`, a cross-run registry of matched-horizon winners
  used by downstream rollout selection. `latest/` remains a convenience alias, not the
  sole winner source.
- Stage-6 selection outputs now include the winning raw endpoint/path MAE plus
  normalized MAE percentages, so the registry remains interpretable without reopening
  `matched_horizon_metrics.csv`.
- Matched-horizon learned candidates now evaluate both `recursive` and
  `direct_endpoint` strategies; `winner_forecast_strategy` captures which strategy
  actually cleared the Stage-6 gates.
- Stage-6 selection is baseline-aware: non-persistence baselines such as `avg_workday`
  can now win a horizon when they clear the same coverage, stability, runtime, and
  practical-gain gates as learned candidates.
- Stage-6 now supports both second-level (`1s`, `5s`, `10s`, `30s`) and minute-level
  (`1min`, `5min`, `10min`, `15min`) modeling cadences. The config-defined execution
  profiles keep smoke/candidate runs bounded while preserving a full-mode path for the
  complete enabled resolution set. Matched-horizon representability is checked in
  exact seconds rather than assuming minute-aligned cadences, so second-level
  resolutions are evaluated instead of being silently skipped.
- Latest smoke run (`outputs/006_multires/commercial_facility/latest_smoke/`, timestamped source `20260309T214557091913Z/`) is green and currently selects:
  - `15m`: learned winner `30s/minimal/ridge-medium/direct_endpoint`
  - `60m`: learned winner `1min/minimal/hgb-balanced/recursive`
- Latest targeted candidate evidence (`outputs/006_multires/20260307T133220706885Z/`) remains mixed:
  - `30m`: learned winner at `5min/curated/hgb-balanced/direct_endpoint`
  - `120m`: learned winner at `5min/curated/hgb-balanced/recursive`
- Latest focused 60-minute tuning run (`outputs/006_multires/commercial_facility/20260310T005916684602Z/`):
  - `60m`: learned winner `5min/minimal/hgb-balanced/recursive`
  - endpoint/path MAE `1148.166851` (`42.691558%`) / `1151.446627` (`36.611950%`)
  - this corrects the earlier stale baseline-led readout
- Stage-7 recursive rollout is executed through
  `scripts/006_recursive_rollout.py` (canonical implementation:
  `scripts/modeling/recursive_rollout.py`) or `run_pipeline.py --stage rollout`, and
  writes timestamped artifacts under `outputs/007_rollout/<artifact_namespace>/` plus a `latest/`
  alias.
- `run_pipeline.py --stage rollout_sweep` executes the Stage-7 challenger sweep and
  writes ranked learned-candidate recommendations under
  `outputs/007_rollout/<load_type>/challenger_sweeps/`.
- Stage-7 now reuses a Stage-6 learned winner only when the requested rollout horizon
  exactly matches the learned winner horizon. It resolves candidates in this order:
  explicit rollout candidate overrides, explicit `--selection-run-id`,
  objective-aware `outputs/007_rollout/<artifact_namespace>/challenger_sweep_registry.csv`,
  `outputs/006_multires/<artifact_namespace>/winner_registry.csv`, legacy `latest/selection_summary.csv`,
  objective-aware `outputs/007_rollout/<artifact_namespace>/rollout_registry.csv`, then
  `config/multires.toml` defaults.
  Direct endpoint winners are not reused for rollout because they do not provide a
  recursive path.
- If any of `--resolution`, `--feature-set`, or `--model-label` are provided, Stage-7
  treats that as an explicit candidate override and does not mix partial CLI input with
  Stage-6 auto-selection.
- Stage-7 now evaluates two additional long-horizon baselines:
  - `anchored_workday`: average-workday shape anchored to the latest observed load level
  - `hybrid_workday`: persistence blended into the anchored-workday path across the horizon
- Stage-7 origin selection is now configurable through `multires.rollout.origin_policy`.
  The repo now supports `uniform`, `midnight`, `billing_aligned`, and
  `phase_balanced`. `phase_balanced` spreads short-horizon origins across the
  full 15-minute phase cycle as a robustness audit. The default rollout and
  horizon-curve configs use `origin_policy=auto`, which resolves from the
  centralized horizon policy in `config/modeling.toml`: `phase_balanced` for
  short/hourly correction horizons and `uniform` for day-ahead horizons.
  `midnight` remains a legacy diagnostic option.
- Stage-7 selection is now objective-aware through `multires.rollout.selection_target`
  (`path_mae`, `endpoint_mae`, `phase_mean_mae`, `next_lock_mae`, or
  `profile_shape_mae`). The default config now uses `selection_target=auto`, which
  resolves to the centralized horizon policy: `next_lock_mae` for short/hourly
  correction horizons and `profile_shape_mae` for day-ahead horizons. The resolved
  target is recorded in `selection_context.json` and used when Stage-7 falls back to
  prior rollout evidence.
- `rollout_selection_summary.csv` and `rollout_selection_summary.md` now record the best
  endpoint, path, and 15-minute phase-average candidate explicitly.
- short-horizon reruns can also emit `rollout_policy_candidates.json`, which records
  any derived phase-bucket policy candidates and the bucket-to-candidate mappings used
  to build them from measured rollout outputs.
- `rollout_registry.csv` records one learned-candidate row per Stage-7 run so long-horizon
  fallback selection is driven by measured rollout outcomes instead of a mutable config row.
- `challenger_sweep_registry.csv` records one recommended challenger row per completed
  Stage-7 sweep so Stage-8 can reuse exact-horizon, exact-origin-policy evidence instead
  of brute-force rerunning every horizon.
- `recursive_rollout_metrics.csv` now includes `phase_mean_mae`, `next_lock_mae`,
  `profile_shape_mae`, and their percentage counterparts so the 15-minute correction
  window and the day-ahead shape can be evaluated directly instead of only through
  endpoint/path proxies.
- Short-horizon rollout policies can now keep multiple residual baselines active
  (`persistence`, `avg_workday`) so Stage-7 can measure whether learned residuals
  close the gap to the strongest baseline family without one-off config edits.
- `recommended_candidate.json` records the best learned challenger at the requested
  rollout horizon after ranking registry-backed candidates against the configured
  selection target.
- Stage-7 challenger sweeps now persist `shared_origins.csv` and force all
  cross-resolution challenger candidates onto the same sampled origin timestamps, so
  sweep recommendations are apples-to-apples instead of origin-sample dependent.
- For hourly horizons, Stage-7 can also synthesize sweep-level
  `portfolio_policy_candidates.json` and `portfolio_policy_by_origin.csv` artifacts
  when different shared-origin challengers win different objectives.
- Sweep-derived portfolio winners can now be replayed as first-class Stage-7 rollout
  runs with `resolution=mixed`, `feature_set=portfolio`,
  `model_label=cross_candidate_portfolio`, plus `portfolio_policy_candidate.json` and
  `shared_origins.csv` so the replay stays auditable against the source sweep.
- `scripts/run_e2e.py --with-rollout` no longer implies `--with-multires`; rollout-only
  verification leaves Stage-6 artifacts untouched unless multires is explicitly
  requested.
- Latest validated `1440m` rollout result is now profile-oriented rather than path-only:
  - challenger recommendation:
    `outputs/007_rollout/commercial_facility/challenger_sweeps/20260310T231235398730Z/recommended_candidate.json`
  - objective: `profile_shape_mae` under `origin_policy=uniform`
  - evaluation scope: `origin_selection_scope=shared_timestamp_intersection`
  - learned winner `10min/minimal/hgb-balanced::raw`
  - learned profile-shape / path / endpoint MAE:
    `717.777613` (`36.245099%`) / `783.077104` (`39.542480%`) /
    `968.909580` (`44.162165%`)
  - persistence profile-shape / path / endpoint MAE:
    `746.527115` (`37.696842%`) / `1010.620668` (`51.032583%`) /
    `1119.137272` (`51.009429%`)
  - best baseline on profile shape is `persistence` at `746.527115` (`37.696842%`)
  - best baseline on path / endpoint remains `avg_workday` at
    `850.145715` (`42.929195%`) / `986.676302` (`44.971959%`)
- Latest validated `15m` autoselection now prioritizes the next correction window:
  - challenger recommendation:
    `outputs/007_rollout/commercial_facility/challenger_sweeps/20260310T234800852734Z/recommended_candidate.json`
  - objective: `next_lock_mae` under `origin_policy=phase_balanced`
  - evaluation scope: `origin_selection_scope=shared_timestamp_intersection`
  - learned winner `1min/minimal_phase/hgb-balanced::phase_bucket_next_lock_policy`
  - learned next-lock / path / phase-average MAE:
    `266.837858` (`9.333856%`) / `266.837858` (`9.333856%`) /
    `167.930779` (`5.874136%`)
  - persistence next-lock MAE: `434.846944` (`15.210731%`)
  - best baseline next-lock MAE: `avg_workday` at `419.789637` (`14.684034%`)
- Latest validated `60m` operational rollout now replays the shared-origin sweep
  winner as a standalone Stage-7 artifact:
  - operational run:
    `outputs/007_rollout/commercial_facility/20260311T015133422915Z/`
  - selection source:
    `outputs/007_rollout/commercial_facility/challenger_sweep_registry.csv`
  - learned winner:
    `cross_candidate_portfolio::phase_bucket_next_lock_policy`
  - learned next-lock / path / profile-shape MAE:
    `253.104260` (`15.969845%`) / `496.893660` (`24.252664%`) /
    `256.446567` (`12.562545%`)
  - persistence next-lock / path / profile-shape MAE:
    `379.116458` (`16.733055%`) / `305.444547` (`14.538286%`) /
    `214.917017` (`10.841634%`)
  - best baseline next-lock MAE: `hybrid_workday` at `375.929657` (`16.483496%`)
- `scripts/run_e2e.py` (canonical implementation: `scripts/tooling/run_e2e.py`) can
  now include the full post-MVP stack in one repository smoke pass via
  `--with-multires --with-rollout --with-rollout-sweep --with-horizon-curve --with-forecast-control`.
- Stage-8 horizon curve is executed through
  `scripts/008_horizon_curve.py` (canonical implementation:
  `scripts/modeling/horizon_curve.py`) or `run_pipeline.py --stage horizon_curve`, and
  writes timestamped artifacts under `outputs/009_horizon_curve/<artifact_namespace>/`
  plus a `latest/` alias.
- Stage-8 consolidates the Stage-5 holdout anchor (`1m`) and the Stage-7 challenger
  sweeps (`15m` through `1440m`) into the H5 capability envelope. This is intentionally
  a horizon-by-horizon best-candidate curve, not a single-model monotonic decay trace.
- Stage-8 now reuses matching rows from
  `outputs/007_rollout/<artifact_namespace>/challenger_sweep_registry.csv` and reruns
  only the horizons whose measured origin policy does not match the requested Stage-8
  policy.
- Key Stage-8 artifacts:
  - `outputs/009_horizon_curve/<artifact_namespace>/horizon_curve_summary.csv`
  - `outputs/009_horizon_curve/<artifact_namespace>/horizon_curve_candidates.csv`
  - `outputs/009_horizon_curve/<artifact_namespace>/horizon_curve_summary.md`
  - `outputs/009_horizon_curve/<artifact_namespace>/crossover_summary.json`
  - `outputs/009_horizon_curve/<artifact_namespace>/fig_horizon_ratio_curve.png`
  - `outputs/009_horizon_curve/<artifact_namespace>/fig_horizon_absolute_mae.png`
- Latest validated Stage-8 readout
  (`outputs/009_horizon_curve/commercial_facility/20260319T163029337069Z/`):
  - `1m` learned superiority is still not supported by the current Stage-5 holdout anchor
    (`175.055450` vs `173.724099`). This Stage-8 row is the current horizon-curve
    characterization point, not the deployable Stage-5 `1m` recommendation.
  - `15m` and `60m` now use `selection_target=next_lock_mae` under
    `origin_policy=phase_balanced`
  - `1440m` now uses `selection_target=profile_shape_mae` under `origin_policy=uniform`
  - reused sweep rows now prefer `origin_selection_scope=shared_timestamp_intersection`
    so post-fix comparable sweeps outrank older non-comparable registry rows
  - `15m`: learned next-lock MAE `266.837858` (`9.333856%`) beats persistence
    `434.846944` (`15.210731%`) and best baseline `419.789637` (`14.684034%`)
  - `60m`: learned next-lock MAE `253.104260` (`15.969845%`) from
    `cross_candidate_portfolio::phase_bucket_next_lock_policy` beats persistence
    `379.116458` (`16.733055%`) and best baseline `375.929657` (`16.483496%`)
  - `1440m`: learned profile-shape MAE `717.777613` (`36.245099%`) beats persistence
    `746.527115` (`37.696842%`) while path MAE `783.077104` also stays ahead of
    `avg_workday` `850.145715`
  - interpretation: the horizon curve is now objective-aware rather than path-only, so
    the repo separately characterizes short-horizon correction quality and day-ahead
    profile quality.

- Stage-10 forecast-control backtest is executed through
  `scripts/009_forecast_control_backtest.py` (canonical implementation:
  `scripts/modeling/forecast_control_backtest.py`) or
  `run_pipeline.py --stage forecast_control`, and writes timestamped artifacts under
  `outputs/010_forecast_control/<artifact_namespace>/` plus a `latest/` alias.
- Stage-10 replays the current measured day-ahead, hourly, and `15m` winners on shared
  24-hour control cycles, then measures whether the intraday correction stack actually
  reduces locked `15m` interval error and profile-shape error versus a frozen day-ahead
  forecast.
- Stage-10 also benchmarks the replayed layer candidates on those exact control cycles
  and writes `control_layer_candidate_benchmarks.csv`, so the backtest can select the
  strongest measured layer candidate instead of assuming the upstream learned label is
  always the operational winner.
- The current latest Stage-10 bundle already includes:
  - calibration vs held-out evaluation control promotion
  - real transition-mismatch refresh triggers
  - denser exact-control `1m` blend search
  - exact-origin replay reuse plus in-process Stage-7 runtime reuse
- Stage-10 now also replays a dedicated day-ahead refresh candidate
  (`hybrid_workday_residual`) at hourly checkpoints and compares:
  - frozen day-ahead
  - unconditional refresh
  - triggered refresh
- Key Stage-10 artifacts:
  - `outputs/010_forecast_control/<artifact_namespace>/control_policy.json`
  - `outputs/010_forecast_control/<artifact_namespace>/control_backtest_summary.csv`
  - `outputs/010_forecast_control/<artifact_namespace>/control_backtest_by_cycle.csv`
  - `outputs/010_forecast_control/<artifact_namespace>/control_minute_timeline.csv`
  - `outputs/010_forecast_control/<artifact_namespace>/control_interval_timeline.csv`
  - `outputs/010_forecast_control/<artifact_namespace>/day_ahead_refresh_summary.csv`
  - `outputs/010_forecast_control/<artifact_namespace>/day_ahead_refresh_decisions.csv`
  - `outputs/010_forecast_control/<artifact_namespace>/phase_stack_guard_summary.csv`
  - `outputs/010_forecast_control/<artifact_namespace>/fig_control_lock_mae.png`
  - `outputs/010_forecast_control/<artifact_namespace>/fig_control_example_cycle.png`
  - `outputs/010_forecast_control/<artifact_namespace>/fig_day_ahead_refresh_policy.png`
- Latest validated Stage-10 readout
  (`outputs/010_forecast_control/commercial_facility/20260322T030301040853Z/`):
  - policy stack after control-cycle benchmarking on all eligible out-of-sample
    validate/test control cycles:
    `10min/minimal/hgb-balanced::raw` ->
    `10min/minimal/hybrid_workday` ->
    `10min/minimal/hybrid_workday` (phase slot via `hourly_passthrough`) ->
    `curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02`
  - upstream challengers were still replayed and benchmarked on the exact control
    cycles:
    `mixed/portfolio/cross_candidate_portfolio::phase_bucket_next_lock_policy` ->
    `10min/minimal/hybrid_workday`,
    `1min/minimal_phase/hgb-balanced::phase_bucket_next_lock_policy` ->
    `1min/minimal_phase_anchor/hgb-balanced::persistence_raw_blend_e25`,
    while the exact stack-aware phase benchmark identified
    `phase_bucket_portfolio::stack_origin_metric_policy` but the broader
    rolling-support guard kept the applied phase slot on hourly passthrough
  - `phase_stack_guard_summary.csv` now records both the exact stack guard and
    the broader rolling-support guard, so the final phase policy is only applied
    when it clears both surfaces
  - the `1m` minute pool now mixes Stage-5 holdout-registry winners, latest
    `blend_finalists.csv`, and remaining raw/residual scoreboard challengers, and
    replays learned `+blend` candidates with their saved Stage-5 blend settings
  - Stage-5 holdout still does not support a blanket learned-superiority claim
  at `1m`, but the exact-control Stage-10 minute surface now promotes a learned
  sparse-feature HGB control-bucket blend over persistence by a large
  operational margin
  - frozen day-ahead lock MAE: `767.411283` (`40.518170%`)
  - after hourly updates lock MAE: `490.428482` (`25.160719%`)
  - after phase updates lock MAE: `490.428482` (`25.160719%`)
  - after nowcast updates lock MAE: `47.503499` (`2.459006%`)
  - frozen day-ahead profile-shape MAE: `788.533702` (`41.185297%`)
  - after hourly updates profile-shape MAE: `626.681554` (`32.175992%`)
  - after phase updates profile-shape MAE: `626.681554` (`32.175992%`)
  - after nowcast updates profile-shape MAE: `174.956343` (`9.038661%`)
  - rolling benchmark adds a broader evidence surface:
    - calibration cycles: `16`
    - evaluation cycles: `16`
    - rolling evaluation lock/profile-shape MAE:
      `763.962699 -> 492.201440 -> 492.201440 -> 47.500033` and
      `786.255244 -> 626.787911 -> 626.787911 -> 175.213594`
    - rolling hourly-vs-day-ahead lock gain:
      `271.761259` with 95% CI [`146.509717`, `379.601589`], `p=0.0000`
    - rolling hourly-vs-day-ahead profile gain:
      `159.467333` with 95% CI [`80.165916`, `228.327102`], `p=0.0000`
    - rolling phase-vs-hourly lock gain:
      `0.000000` with 95% CI [`0.000000`, `0.000000`], `p=1.0000`
    - rolling nowcast-vs-phase lock gain:
      `444.701407` with 95% CI [`408.156351`, `482.162622`], `p=0.0000`
  - day-ahead refresh study:
    - refresh candidate: `10min/minimal/hgb-balanced::hybrid_workday_residual`
    - frozen day-ahead profile-shape MAE: `788.533702` (`41.185297%`)
    - unconditional refresh profile-shape MAE: `701.862380` (`36.420548%`)
    - triggered refresh profile-shape MAE: `732.516445` (`37.782077%`)
    - frozen day-ahead lock MAE: `767.411283` (`40.518170%`)
    - unconditional refresh lock MAE: `606.603723` (`31.678078%`)
    - triggered refresh lock MAE: `655.385169` (`33.805756%`)
    - mean triggered refresh updates applied per cycle: `8.75`
    - evaluation trigger rate: `0.3804347826`
    - rolling trigger rate: `0.3838028169`
    - selected trigger mode: `residual_or_activity_active_or_transition`
    - triggered refresh preserved `64.63%` of the unconditional
      profile-shape gain and `69.66%` of the unconditional lock gain
    - rolling benchmark also recommends `triggered_refresh`:
      triggered refresh lock/profile-shape MAE `649.485955` / `730.181648`
    - current trigger reason mix is now split between residual drift and
      activity-profile shift
    - current promoted operating mode is therefore `triggered_refresh`
  - latest focused standalone `1440m` sweep
    (`outputs/007_rollout/commercial_facility/challenger_sweeps/20260320T090013545419Z/`):
    - `10min/minimal/hgb-balanced::raw` remains the best standalone `24h` rollout
      at profile-shape MAE `717.777613` (`36.245099%`)
    - `10min/minimal/hgb-balanced::hybrid_workday_residual` is weaker as a
      standalone full-day rollout than the frozen anchor and remains more useful
      as a refresh path than as the primary `24h` rollout
    - interpretation: the residual model is currently a refresh path, not the
      primary frozen day-ahead anchor
  - replay cache registry:
    `outputs/010_forecast_control/commercial_facility/replay_cache/replay_cache_registry.csv`
  - current evidence index:
    `outputs/010_forecast_control/commercial_facility/latest/current_evidence_index.md`
  - interpretation: the repo now has a fully persisted four-layer control
    readout under the stricter methodology and broader all-eligible out-of-sample
    replay. The hourly layer is clearly useful, the stack-applied phase layer is
    now learned and rolling-positive, and the final `1m` layer is currently a
    learned adaptive-HGB nowcast on the exact-control surface even though Stage-5
    holdout still favors persistence.

  > **Note:** The figures below are generated by running Stage-10 and are not
  > stored in version control.

  ![Stage-10 locked-interval MAE progression](../../outputs/010_forecast_control/commercial_facility/latest/fig_control_lock_mae.png)

  ![Stage-10 example control cycle](../../outputs/010_forecast_control/commercial_facility/latest/fig_control_example_cycle.png)

  ![Stage-10 day-ahead refresh policy comparison](../../outputs/010_forecast_control/commercial_facility/latest/fig_day_ahead_refresh_policy.png)

Latest verification snapshot:
- `python scripts/validate_notebooks.py` -> success; latest archive
  `outputs/008_notebook_runs/commercial_facility/20260322T034149455817Z/`
- `python scripts/write_validation_snapshot.py` -> success; canonical current-state page
  `docs/003_modeling/current_validation_snapshot.md`
- `pytest -q` -> success (`248.47s`)
- `python scripts/run_e2e.py --mode quick --with-multires --with-rollout --with-rollout-sweep --with-horizon-curve --with-forecast-control`
  -> success (`2655.07s` total; pipeline `1951.19s`, notebooks `455.41s`,
  pytest `248.47s`)
