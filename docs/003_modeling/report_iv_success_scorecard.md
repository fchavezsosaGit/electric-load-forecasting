# Report IV Success Scorecard (Codebase Reality Check)

> Historical scope note:
> This document preserves the Report IV success framing. Current operating truth
> is governed by
> [current_validation_snapshot.md](current_validation_snapshot.md),
> [current_operating_approach.md](current_operating_approach.md), and
> [002_operating_direction_spec.md](../000_governance/002_operating_direction_spec.md).

This scorecard maps the success framework in `personal/success.md` to the current
repository state and latest executed artifacts.

Canonical latest-state reference:
- use the validation-command block below plus
  [`current_validation_snapshot.md`](current_validation_snapshot.md) for the
  canonical one-page current-state summary
- use
  [`current_operating_approach.md`](current_operating_approach.md) for the
  current operating stack and timing interpretation
- use [model_and_blend_guide.md](model_and_blend_guide.md) when current labels
  or wrapper names are unfamiliar
- older bundle ids that still appear later in this scorecard are historical
  evidence unless they are explicitly restated in the validation block

Date: 2026-03-19  
Scope: current codebase + latest integrated quick E2E + standalone notebook validation + full pytest + targeted Stage-5 GPU probe  
Primary evidence: `outputs/004_modeling/commercial_facility/*`,
`outputs/005_performance/commercial_facility/latest/*`,
`outputs/006_multires/commercial_facility/latest/*`,
`outputs/007_rollout/commercial_facility/latest/*`,
`outputs/008_notebook_runs/commercial_facility/20260322T042828995952Z/*`,
`outputs/009_horizon_curve/commercial_facility/latest/*`,
`outputs/010_forecast_control/commercial_facility/latest/*`, pipeline outputs in
`data/*`, and validation commands listed below.

---

## Attribution (Implementation + QA)

To accurately reflect ownership:

- **Team contribution (~70%)**: project framing, medallion architecture, hypothesis and
  MVP scope decisions, baseline implementation path, and primary modeling strategy.
- **AI support contribution (~30%)**: gap identification, consistency hardening,
  validation guardrails, and documentation synchronization.

This scorecard uses that framing and does not re-attribute core project direction.

---

## Validation Commands Executed

- `python scripts/validate_notebooks.py`
- `python -m pytest -q`
- `python scripts/run_e2e.py --mode quick --with-multires --with-rollout --with-rollout-sweep --with-horizon-curve --with-forecast-control`

All runtime and test commands completed successfully in this review pass.
Latest integrated quick E2E timing snapshot:
- pipeline: `1951.19s`
- notebooks: `455.41s`
- pytest: `248.47s`
- total: `2655.07s`

Latest standalone notebook-validation evidence:
- archive run: `outputs/008_notebook_runs/commercial_facility/20260322T042828995952Z/`
- modeling artifact validation: `57` `metrics_overall.csv` rows, `98`
  `metrics_by_day_class.csv` rows, `1176` `metrics_by_hour.csv` rows
- validated modeling figures:
  `1990x772`, `1792x814`, `1966x1134`, `1990x772`

Latest standalone pytest timing snapshot:
- `python -m pytest -q`: `248.47s`

---

## Stage-by-Stage Success Status

| Stage | Success Definition (from `success.md`) | Current Evidence | Status |
|---|---|---|---|
| Project-level | Reproducible end-to-end pipeline + validated modeling path | Integrated quick E2E, notebook validation, and full pytest all succeeded in the current pass | PASS |
| Raw | Correct `.mat` structure and preservation | `P_data` shape `(86400, 31)`, `day_data` `(1, 31)`, `day_class` `(1, 31)`, NaN count `14576` | PASS |
| Bronze | Lossless long-format conversion with stable schema | `2,678,400` rows, correct timestamp span, NaN load count preserved | PASS |
| Silver | 89-column feature engineering across default resolutions | `1m/5m/10m/15m` all generated, each `89` columns | PASS |
| Gold | Null-safe modeling contract on required core columns | Gold outputs have `avg_load` null count `0` in all default resolutions | PASS |
| Model datasets | Chronological split outputs for all feature sets/resolutions | `96` parquet files are present in `data/004_model/` (`4 resolutions x 8 feature sets x 3 splits`) | PASS |
| Exploratory Data Analysis (EDA) notebooks | Executable notebooks with centralized config behavior | Full `validate_notebooks.py` completed (000-003 notebooks), archived run `20260322T042828995952Z`, and validated the four `003_modeling.ipynb` figures plus three CSV outputs | PASS |
| Modeling MVP | 1min fixed grid + baselines + holdout protocol | `57` experiment rows plus one-shot holdout and day-ahead extension are present in `outputs/004_modeling/commercial_facility/` | PASS |
| Stage-5 Performance | Preflight + fold evaluation + residual + regularization + blend guardrail | `outputs/005_performance/commercial_facility/20260320T101213998808Z/` contains the latest promoted challenger plus coverage, segmented holdout, deployment recommendation, bootstrap holdout inference, permutation importance, `blend_finalists.csv`, and the cross-run `holdout_registry.csv`; current operational recommendation remains persistence-led | PASS |
| Hypothesis execution | H1/H2/H4 evaluated in-notebook; H3 implemented in Stage-6 runtime; H5 implemented in Stage-8 | H1/H2/H4 rows present; H3 notebook row remains deferred while Stage-6 multires runs are operational; H5 now has a generated horizon curve with explicit cross-horizon winners and failures | PASS |
| Forecast-control backtest | Current 24h forecast + intraday correction stack is replayed end-to-end on shared control cycles, with replayed layer candidates benchmarked on the exact control window, repeated runs reusing exact-origin replay cache artifacts, a dedicated day-ahead refresh study comparing frozen versus refreshed `24h` profiles, a stack-level phase benchmark that can promote a stronger `15m` stack policy than the isolated winner, and a rolling benchmark across all eligible out-of-sample validate/test control cycles on the configured schedules | `outputs/010_forecast_control/commercial_facility/20260322T030301040853Z/` is the current fully persisted Stage-10 bundle and shows exact-control lock MAE `767.411283 -> 490.428482 -> 490.428482 -> 47.503499`; the current control stack is learned at day-ahead, baseline-led at hourly, hourly passthrough at the structural `15m` phase slot after a stack-level veto, and learned at `1m` on the exact-control surface | PASS |
| Evaluation integrity | Validation-driven selection; holdout not used for tuning | Coverage-guard selection policy serialized in `run_manifest.json` | PASS |
| Infra/testing | Passing tests + operational commands | `pytest -q: success`, notebook validation + pipeline runs succeeded | PASS |
| Documentation/governance | Current docs aligned to current behavior | Architecture/modeling/glossary/report docs refreshed in this pass | PASS |

---

## Modeling Performance Snapshot (Latest Run)

Source: `outputs/004_modeling/commercial_facility/metrics_overall.csv`,
`outputs/004_modeling/commercial_facility/run_manifest.json`

### Validation

- Raw MAE-best validate model: `full_stable/ridge/strong`
  - MAE `502.5372` (`13.1423%`), RMSE `949.1225` (`24.8214%`)
  - Coverage `1612 / 4296 = 0.3752` (low-coverage caveat)
- Coverage-selected holdout candidate: `curated/ridge/strong`
  - MAE `596.7663` (`15.0504%`), RMSE `1141.1234` (`28.7790%`)
  - Coverage `4152 / 4296 = 0.9665`
- Validate persistence baseline: MAE `594.7170` (`14.5976%`), RMSE `1195.9468` (`29.3550%`)

Improvement versus persistence:

- Raw-best vs persistence (MAE): `+15.50%`
- Coverage-selected vs persistence (MAE): `-0.34%`

### Holdout (One-shot Test)

- Canonical notebook holdout model (`curated/ridge/strong`) test MAE: `213.2404` (`10.2868%`)
- Canonical notebook persistence holdout MAE: `173.7241` (`8.3805%`)
- Latest promoted Stage-5 holdout challenger (`curated_ramp/xgb-balanced/residual+blend`) MAE: `175.0555` (`8.4447%`)
- Latest promoted Stage-5 holdout challenger RMSE: `268.8967` (`12.9717%`)
- Latest promoted Stage-5 challenger loses to persistence on MAE by `1.3314 W`
- MAE delta 95% moving-block bootstrap CI: `[+0.4013, +2.2254] W`
- One-sided p-value for the claim "learned MAE < persistence MAE": `1.0000`
- Exact conclusion: `1m` learned superiority is not supported by the current holdout evidence
- RMSE delta versus persistence: `-2.0603 W` with 95% CI `[-3.1503, -0.9423] W`
- Added secondary smart baselines: `arima` MAE `863.1136` (`41.6369%`) and `holt_damped` MAE `864.4399` (`41.7008%`)
- Top learned-challenger feature signal remains concentrated: `lag_1` remains the top feature and the top 5 features contribute `94.19%` of positive permutation importance
- Best historical Stage-5 learned holdout winner preserved in `holdout_registry.csv`:
  `full_stable/hgb-frontier-lr010-l2001/raw+blend` at `163.0038` (`7.8634%`)
- Validate-to-test MAE shift (notebook-selected model): `-64.27%`

### Day-ahead Extension

- `experiment=day_ahead` row present
- MAE `2665.8329` (`65.4041%`)
- Coverage `4296 / 4296 = 1.0000`

---

## Hypothesis Status Snapshot

- H1 (`h1_control` vs `minimal/ridge/medium`, MAE): `+4.01%` (improvement, below `>=10%` target)
- H2 (`temporal/ridge/medium` vs `curated/ridge/medium`, RMSE): `+5.27%` (improvement, below `>=8%` target)
- H3: canonical notebook row remains deferred by MVP scope, but the Stage-6 multiresolution runtime is implemented for matched-horizon evaluation and now has learned winners at `30m`, `60m`, and `120m`; the latest smoke run keeps persistence at `15m`, and `60m` remains profile-sensitive.
- H4 (exploratory): Ridge still beats the comparable HGB variants on the curated validate readout in this run
- H5 (horizon degradation): Stage-8 horizon curve is now executed. The current objective-aware envelope separates short-horizon next-lock quality from day-ahead profile quality: `15m` and `60m` beat persistence on next-lock MAE, `1440m` beats persistence on profile-shape MAE, and the remaining loss is the `1m` holdout anchor

Interpretation: implementation integrity is strong; hypothesis outcomes remain mixed.
The repo now has a much stronger challenger-selection surface than the notebook alone.
Stage-5 still favors persistence at the `1m` holdout, while the current cross-horizon
surface shows learned wins at `15m`, `60m`, and `1440m` only when the metric is
matched to the horizon objective.

### Post-MVP Runtime Snapshot

- Latest integrated quick E2E:
  - `python scripts/run_e2e.py --mode quick --with-multires --with-rollout --with-rollout-sweep --with-horizon-curve --with-forecast-control`
  - pipeline `1536.31s`, notebooks `304.44s`, pytest `117.53s`, total `1958.28s`
- Stage-6 candidate result is mixed:
  - `latest_smoke` winner at `matched_horizon_15m`: baseline persistence
  - `latest_smoke` winner at `matched_horizon_60m`: learned model `1min/minimal/hgb-balanced/recursive`
  - `matched_horizon_30m` winner: learned model at `5min/curated/hgb-balanced/direct_endpoint`
  - `matched_horizon_60m` focused winner: learned model at `5min/minimal/hgb-balanced/recursive`
    with endpoint/path MAE `1148.166851` (`42.691558%`) / `1151.446627` (`36.611950%`)
  - `matched_horizon_120m` winner: learned model at `5min/curated/hgb-balanced/recursive`
  - interpretation: learned candidates now clear the configured gates at several
    matched horizons, but `60m` remains profile-sensitive and still favors `minimal`
    over richer `regime_profile` and `full_stable` challengers in the latest rerun.
- Stage-6 `resolution_health.csv` contains four learned rows marked
  `coverage_below_threshold`; that is expected gate behavior for ineligible candidates,
  not a stage failure. The run manifest status remains `success`.
- Stage-7 rollout selection is now horizon-aware:
  - a Stage-6 learned winner is reused only when it matches the requested rollout horizon
    and it is a `recursive` winner rather than an endpoint-only direct strategy
  - resolution order is now explicit rollout candidate overrides, explicit
    `--selection-run-id`, Stage-7 `challenger_sweep_registry.csv`, Stage-6 `winner_registry.csv`, legacy
    `latest/selection_summary.csv`, Stage-7 `rollout_registry.csv`, then config fallback
  - fallback selection is now objective-aware through `selection_target`
  - otherwise the rollout falls back to the explicit config candidate and records that
    reason in `selection_context.json`
- Stage-7 challenger selection is now explicit through
  `python run_pipeline.py --stage rollout_sweep`, which ranks learned rollout
  candidates using `rollout_registry.csv` evidence and writes
  `recommended_candidate.json`.
- Stage-7 challenger sweeps now evaluate every cross-resolution candidate on the same
  sampled origin timestamps and persist `shared_origins.csv`, so sweep promotion is no
  longer origin-sample dependent.
- Sweep-derived portfolio winners can now be replayed as standalone Stage-7 rollout
  runs with `resolution=mixed`, `feature_set=portfolio`, and
  `model_label=cross_candidate_portfolio`.
- Latest validated `1440m` rollout result is now profile-oriented:
  - challenger recommendation:
    `outputs/007_rollout/commercial_facility/challenger_sweeps/20260310T231235398730Z/recommended_candidate.json`
  - objective `profile_shape_mae` under `origin_policy=uniform`
  - evaluation scope `origin_selection_scope=shared_timestamp_intersection`
  - learned winner `10min/minimal/hgb-balanced::raw`
  - learned profile-shape / path / endpoint MAE:
    `717.777613` (`36.245099%`) / `783.077104` (`39.542480%`) /
    `968.909580` (`44.162165%`)
  - persistence profile-shape / path / endpoint MAE:
    `746.527115` (`37.696842%`) / `1010.620668` (`51.032583%`) /
    `1119.137272` (`51.009429%`)
- Stage-8 horizon curve (`outputs/009_horizon_curve/commercial_facility/20260312T065037307284Z/`)
  now provides the actual H5 characterization surface:
  - `15m`: learned next-lock MAE `266.837858` (`9.333856%`) beats persistence
    `434.846944` (`15.210731%`) and best baseline `419.789637` (`14.684034%`)
  - `60m`: learned next-lock MAE `253.104260` (`15.969845%`) from
    `cross_candidate_portfolio::phase_bucket_next_lock_policy` beats persistence
    `379.116458` (`16.733055%`) and best baseline `375.929657` (`16.483496%`)
  - `1440m`: learned profile-shape MAE `717.777613` (`36.245099%`) beats persistence
    `746.527115` (`37.696842%`)
  - remaining weak point is still the `1m` holdout anchor
- Latest operational `60m` rollout replay is now
  `outputs/007_rollout/commercial_facility/20260311T015133422915Z/`, which preserves
  the same `cross_candidate_portfolio::phase_bucket_next_lock_policy` winner as a
  normal Stage-7 run instead of leaving it only in the sweep namespace.
- Stage-10 forecast-control backtest
  (`outputs/010_forecast_control/commercial_facility/20260322T030301040853Z/`) now
  benchmarks the replayed layer candidates on the exact control window, reuses
  replay-cache artifacts, and replays all eligible out-of-sample validate/test
  control cycles on the configured schedules:
  - day-ahead / hourly / phase / nowcast policies:
    `10min/minimal/hgb-balanced::raw` ->
    `10min/minimal/hybrid_workday` ->
    `10min/minimal/hybrid_workday` (phase slot via `hourly_passthrough`) ->
    `curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02`
  - the exact phase-stack benchmark still identified
    `phase_bucket_portfolio::stack_origin_metric_policy`, but the broader
    rolling-support guard vetoed it for current operation
  - exact-control coverage:
    - calibration cycles: `8`
    - evaluation cycles: `8`
  - frozen day-ahead lock MAE: `767.411283` (`40.518170%`)
  - after hourly updates: `490.428482` (`25.160719%`)
  - after phase updates: `490.428482` (`25.160719%`)
  - after nowcast updates: `47.503499` (`2.459097%`)
  - frozen day-ahead profile-shape MAE: `788.533702` (`41.185297%`)
  - after hourly updates profile-shape MAE: `626.681554` (`32.175992%`)
  - after phase updates profile-shape MAE: `626.681554` (`32.175992%`)
  - after nowcast updates profile-shape MAE: `174.956343` (`9.038730%`)
  - rolling benchmark now broadens the control evidence:
    - calibration cycles: `16`
    - evaluation cycles: `16`
    - evaluation lock/profile-shape MAE:
      `763.962699 -> 492.201440 -> 492.201440 -> 47.500033` and
      `786.255244 -> 626.787911 -> 626.787911 -> 175.213594`
    - hourly-vs-day-ahead lock gain:
      `271.761259` with 95% CI [`146.509717`, `379.601589`]
    - hourly-vs-day-ahead profile gain:
      `159.467333` with 95% CI [`80.165916`, `228.327102`]
    - phase-vs-hourly lock gain:
      `0.000000` with 95% CI [`0.000000`, `0.000000`]
    - nowcast-vs-phase lock gain:
      `444.701407` with 95% CI [`408.156351`, `482.162622`]
  - day-ahead refresh study:
    `10min/minimal/hgb-balanced::hybrid_workday_residual` improves the frozen
    day-ahead path when always applied, and the current promoted operating mode
    is now `triggered_refresh` because the trigger preserves enough of that
    unconditional gain while staying inside the configured trigger-rate band:
    - unconditional refresh lock/profile-shape MAE:
      `606.603723` (`31.678078%`) / `701.862380` (`36.420548%`)
    - triggered refresh lock/profile-shape MAE:
      `655.385169` (`33.752357%`) / `732.516445` (`37.704136%`)
    - selected trigger mode: `residual_or_activity_active_or_transition`
    - trigger rate:
      exact `0.3804347826`, rolling `0.3838028169`
    - triggered refresh kept `64.63%` of the unconditional profile gain and
      `69.66%` of the unconditional lock gain
    - rolling benchmark also recommends `triggered_refresh`:
      `649.485955` / `730.181648`
  - focused standalone `1440m` sweep:
    `outputs/007_rollout/commercial_facility/challenger_sweeps/20260320T090013545419Z/`
    still prefers standalone `10min/minimal/hgb-balanced::raw` at profile-shape
    MAE `717.777613` (`36.245099%`), which means `hybrid_workday_residual` is
    currently a refresh path rather than the best frozen `24h` anchor
  - the isolated learned `15m` winner is still useful, but the current
    stack-applied phase slot is hourly passthrough because broader rolling
    support did not justify a distinct `15m` correction
  - `1m` learned superiority is not supported on the current Stage-5
    holdout/control evidence, but the final exact-control nowcast layer is now
    learned and GPU-capable
  - exact-origin replay cache registry:
    `outputs/010_forecast_control/commercial_facility/replay_cache/replay_cache_registry.csv`
  - current evidence index:
    `outputs/010_forecast_control/commercial_facility/latest/current_evidence_index.md`

---

## Remaining Risk (Not a Code Bug)

Pipeline-level analytical caveat:

- `rolling_*_1440` NaN propagation can reduce Ridge evaluation coverage for `full`
  feature set, creating non-uniform comparison subsets.

Current mitigation already implemented:

- Coverage-guard holdout selection (`MIN_VALIDATE_COVERAGE=0.95`)
- Manifest records both raw-best and selected-best for audit transparency.

---

## Definition-of-Success Readout

Against the success framework, the codebase is currently in a **deliverable-ready**
state for Report IV execution:

- Pipeline correctness and reproducibility: achieved
- Modeling/holdout protocol integrity: achieved
- Documentation and references: updated to current implementation
- Remaining work is primarily model-improvement iteration, not infrastructure repair
