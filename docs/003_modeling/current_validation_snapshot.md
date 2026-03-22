# Current Validation Snapshot

This file is generated from the latest persisted artifacts. It is the
canonical answer to "what is the repo's current validated state?"

- Generated at: `2026-03-22T05:10:19.128840+00:00`
- Artifact namespace: `commercial_facility`

## Validation Surface

- Stage-5 latest manifest: [`../../outputs/005_performance/commercial_facility/latest/run_manifest.json`](../../outputs/005_performance/commercial_facility/latest/run_manifest.json)
- Stage-6 latest manifest: [`../../outputs/006_multires/commercial_facility/latest/run_manifest.json`](../../outputs/006_multires/commercial_facility/latest/run_manifest.json)
- Stage-7 latest rollout manifest: [`../../outputs/007_rollout/commercial_facility/latest/run_manifest.json`](../../outputs/007_rollout/commercial_facility/latest/run_manifest.json)
- Stage-7 latest sweep manifest: [`../../outputs/007_rollout/commercial_facility/challenger_sweeps/latest/run_manifest.json`](../../outputs/007_rollout/commercial_facility/challenger_sweeps/latest/run_manifest.json)
- Stage-8 latest manifest: [`../../outputs/009_horizon_curve/commercial_facility/latest/run_manifest.json`](../../outputs/009_horizon_curve/commercial_facility/latest/run_manifest.json)
- Stage-10 latest manifest: [`../../outputs/010_forecast_control/commercial_facility/latest/run_manifest.json`](../../outputs/010_forecast_control/commercial_facility/latest/run_manifest.json)
- Notebook archive manifest: [`../../outputs/008_notebook_runs/commercial_facility/latest/run_manifest.json`](../../outputs/008_notebook_runs/commercial_facility/latest/run_manifest.json)
- Stage-5 holdout coverage summary: [`../../outputs/005_performance/commercial_facility/latest/holdout_coverage_summary.json`](../../outputs/005_performance/commercial_facility/latest/holdout_coverage_summary.json)
- Stage-5 supplemental surface advisory: [`../../outputs/005_performance/commercial_facility/latest/supplemental_surface_advisory.json`](../../outputs/005_performance/commercial_facility/latest/supplemental_surface_advisory.json)
- Stage-10 runtime summary: [`../../outputs/010_forecast_control/commercial_facility/latest/runtime_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/runtime_summary.json)
- Exact-control cycles: calibration `8`, evaluation `8`
- Rolling benchmark cycles: calibration `16`, evaluation `16`

## Current Findings

- Stage-5 holdout recommendation:
  `persistence`
  because persistence remains the operational winner because the promoted Stage-5 candidate did not beat the strongest baseline on holdout MAE.
- Stage-5 minute operating policy:
  standalone `baseline_anchor` / Stage-10 `corrective_overlay_specialist`
- Best current learned 1m challenger:
  `curated_ramp/xgb-balanced/residual+blend` at 175.055450 (8.444727%) vs persistence 173.724099 (8.380502%)
- Stage-5 holdout coverage note:
  The promoted Stage-5 holdout surface covers only one observed value for at least one key operating segment, so standalone 1-minute claims should be read as narrow-regime evidence.
- Stage-5 supplemental advisory surface:
  learned beats persistence on the broader advisory surface: `True`; learned-supported operating regimes: `transition_active, transition_only`
- Strongest supplemental diagnostic segment:
  actual_ramp_band=high_ramp at ratio `0.878199` over `862` rows
- Stage-7 latest day-ahead sweep recommendation:
  `10min/minimal/hgb-balanced::raw` on `profile_shape_mae` with 717.777613 (36.245099%)
- Stage-8 objective winners:
- `1m`: `curated_ramp/xgb-balanced/residual+blend` on `endpoint_mae` with 175.055450 vs persistence 173.724099
- `15m`: `hgb-balanced::phase_bucket_next_lock_policy` on `next_lock_mae` with 266.837858 vs persistence 434.846944
- `60m`: `cross_candidate_portfolio::phase_bucket_next_lock_policy` on `next_lock_mae` with 253.104260 vs persistence 379.116458
- `1440m`: `hgb-balanced::raw` on `profile_shape_mae` with 717.777613 vs persistence 746.527115

## How To Read The Winners

- Stage-5 answers the deployable `1m` holdout question. If it keeps `persistence`, that is the honest short-horizon recommendation.
- Stage-8 answers the horizon-characterization question. Its `1m` row does not override the Stage-5 deployment recommendation by itself.
- Stage-10 answers the control-stack question. It may choose a different nowcast layer after replaying candidates on the exact control cycles.
- Candidate-label anatomy, blend wrappers, and the current CPU/GPU policy are summarized in [model_and_blend_guide.md](model_and_blend_guide.md).

## Current Resolution Policy

- Supported pipeline resolutions: `1s`, `5s`, `10s`, `30s`, `1min`, `5min`, `10min`, `15min`
- Default materialized pipeline resolutions: `1min`, `5min`, `10min`, `15min`
- Current optimizer-facing actual resolution: `1min` with `15` minute lock intervals.
- Latest matched-horizon winners that cleared the Stage-6 gates: `15m` -> `1min` (baseline model), `60m` -> `1min` (learned model)
- Best current sub-minute challenger: `30s/minimal/xgb-balanced/direct_endpoint` at `60m` with MAE ratio `0.653587` to persistence.
- Why that does not replace `1min` today: the best current sub-minute candidate still fails the Stage-6 operating gates, so it stays exploratory instead of becoming the default control cadence.
- Current gate readout for that challenger: `eligible=False` / `pareto_passed=False` / `practical_gain_passed=True` / `fold_std_mae_ratio=0.281089` against the configured stability gate `0.200000`.
- Current operating rule: keep `1min` as the validated optimizer-facing correction cadence, treat `30s` as exploratory where it shows raw matched-horizon promise, and do not promote `1s` / `5s` / `10s` without new persisted evidence that they beat the current gates.
- Practical implication: the repo can ingest and compare sub-minute data, but the current Stage-10 delivery contract is still a `1min` nowcast overlay on top of the broader `15m` decision loop.

## Current Control Stack

- Day-ahead: `10min/minimal/hgb-balanced::raw`
- Hourly: `10min/minimal/hybrid_workday`
- Stack-applied phase: `phase_bucket_portfolio::stack_origin_metric_policy`
- Phase slot note: the current applied phase slot uses `phase_bucket_portfolio::stack_origin_metric_policy`
- Nowcast: `curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02`

| Layer | Lock MAE | Profile-Shape MAE |
|-------|----------|-------------------|
| Frozen day-ahead | `767.411283` | `788.533702` |
| After hourly updates | `490.428482` | `626.681554` |
| After phase updates | `417.229872` | `570.445008` |
| After nowcast updates | `47.503499` | `174.956343` |

- Rolling evaluation stack:
  lock `763.962699` -> `492.201440` -> `270.220472` -> `47.500033`
  profile `786.255244` -> `626.787911` -> `432.403931` -> `175.213594`

## Optimizer Delivery Surface

- Delivery contract: [`../../outputs/010_forecast_control/commercial_facility/latest/optimizer_delivery_contract.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_delivery_contract.json)
- Operational policy: [`../../outputs/010_forecast_control/commercial_facility/latest/optimizer_operational_policy.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_operational_policy.json)
- Stage-5 minute operating policy: [`../../outputs/005_performance/commercial_facility/latest/operating_policy.json`](../../outputs/005_performance/commercial_facility/latest/operating_policy.json)
- Delivery preview: [`../../outputs/010_forecast_control/commercial_facility/latest/optimizer_delivery_preview.csv`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_delivery_preview.csv)
- Dynamic overlay shadow summary: [`../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_shadow_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_shadow_summary.json)
- Soft overlay shadow summary: [`../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_soft_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_soft_summary.json)
- Uncertainty summary: [`../../outputs/010_forecast_control/commercial_facility/latest/optimizer_delivery_uncertainty_summary.csv`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_delivery_uncertainty_summary.csv)
- Contract version: `1.2`
- Delivery cadence: `15` minutes
- Layer priority: `nowcast, phase, hourly, day_ahead`
- Uncertainty method: `contextual_empirical_residual_quantiles`
- Confidence signal: `heuristic_operational_trust_score`
- Preview rows carry freshness fields: `as_of_timestamp, effective_forecast_as_of, expected_layer_cadence_minutes, forecast_age_minutes, stale_threshold_minutes, is_stale_forecast`
- Runtime portability: `CPU-safe HGB and baseline paths remain the default-safe contract for non-accelerated and ARM64 hosts.`
- Minute-layer operating role: `corrective_overlay_specialist`
- Dynamic minute-overlay recommendation: `keep_shadow_mode`
  shadow all-interval abs error `47.503499` vs enforced `356.453113`; delta `308.949614`
- Soft minute-overlay shadow recommendation: `keep_pure_nowcast_shadow`
  best policy `soft_overlay_sw100_bw100` at all-interval abs error `47.503499`
- All-interval 80% / 95% empirical coverage: `0.934278` / `0.984536`
- Stage-10 runtime hotspot: `select_phase_stack_policy` at `174.917094` seconds

## Day-Ahead Refresh

- Recommended policy: `triggered_refresh`
- Trigger mode: `residual_or_activity_active_or_transition`
- Exact-control trigger rate: `0.380435`
- Rolling trigger rate: `0.383803`
- Exact frozen/unconditional/triggered profile MAE: `788.533702` / `701.862380` / `732.516445`

## Notebook Evidence

- Notebook archive status: `success`
- `003_modeling.ipynb` output count: `50`
- `metrics_overall.csv` rows: `63`
- `metrics_by_day_class.csv` rows: `110`
- `metrics_by_hour.csv` rows: `1320`

## Interpretation

- The full quick validation surface is currently green.
- The repo still does not support a learned-superiority claim at `1m`.
- Stage-5 now persists an explicit minute operating policy so the repo can say, in writing, that standalone `1m` stays baseline-led while Stage-10 may still use learned minute overlays as corrective specialists.
- The broader leakage-safe Stage-5 supplemental surface now shows where learned `1m` value actually appears: the latest advisory run beats persistence overall on stitched validate-walkforward plus holdout rows, but that support is concentrated in transition regimes rather than the narrow canonical holdout slice.
- The layered stack now carries a dynamic minute-overlay controller in shadow mode. That is intentional: the repo persists both a hard-gate shadow-vs-enforced counterfactual and a soft-overlay shadow search so adaptive minute routing can prove itself on the Stage-10 surface before it is allowed to change live layer resolution.
- The current stack-applied `15m` phase layer now adds a meaningful rolling gain on top of the hourly layer, while the final minute nowcast remains the largest improvement.
- The phase stack guard now checks next-lock and peak behavior explicitly, so a phase layer must clear optimizer-relevant guardrails instead of winning only on broader lock/profile metrics.
- Stage-10 now also emits a pre-optimizer interval contract with calibrated residual bands, freshness fields, and confidence hints, so the repo can expose forecast rows with timestamps, horizons, fallback context, and uncertainty instead of only aggregate replay metrics.
- The Stage-7 sweep is now part of the normal E2E verification path, so the repo's 'full pass' no longer skips that surface.

## High-Signal Visual Anchors

- Horizon capability curve:
  ![Stage-8 horizon ratio curve](../../outputs/009_horizon_curve/commercial_facility/latest/fig_horizon_ratio_curve.png)
- Control-layer gain surface:
  ![Stage-10 control layer gain curve](../../outputs/010_forecast_control/commercial_facility/latest/fig_control_layer_gain_ci.png)

## Visualization Surfaces

- Use the integrated dashboard when you want the latest cross-stage visual story in one place.
- [Current Visualization Guide](current_visualization_guide.md)
- [`../../outputs/reports/commercial_facility/latest/validation_dashboard.html`](../../outputs/reports/commercial_facility/latest/validation_dashboard.html)

Supporting references:
- [Current Operating Approach](current_operating_approach.md)
- [Model and Blend Guide](model_and_blend_guide.md)
- [README](../../README.md)
- [`../../outputs/010_forecast_control/commercial_facility/latest/current_evidence_index.md`](../../outputs/010_forecast_control/commercial_facility/latest/current_evidence_index.md)
