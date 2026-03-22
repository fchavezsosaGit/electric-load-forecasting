# Optimizer Delivery Contract

This document describes the interval-level forecast surface that Stage-10 now
emits for pre-optimizer consumption.

The API layer is still out of scope. This contract exists so the repo can
answer a simpler question first:

If another system asked us for the current forecast/nowcast feed, what exact
fields would we hand it, and what evidence would justify trusting them?

Primary runtime artifacts:
- `outputs/010_forecast_control/<artifact_namespace>/latest/optimizer_delivery_contract.json`
- `outputs/010_forecast_control/<artifact_namespace>/latest/optimizer_operational_policy.json`
- `outputs/010_forecast_control/<artifact_namespace>/latest/optimizer_delivery_preview.csv`
- `outputs/010_forecast_control/<artifact_namespace>/latest/optimizer_delivery_serving_preview.csv`
- `outputs/010_forecast_control/<artifact_namespace>/latest/optimizer_dynamic_overlay_shadow_summary.json`
- `outputs/010_forecast_control/<artifact_namespace>/latest/optimizer_dynamic_overlay_soft_summary.json`
- `outputs/010_forecast_control/<artifact_namespace>/latest/optimizer_dynamic_overlay_soft_candidates.csv`
- `outputs/010_forecast_control/<artifact_namespace>/latest/optimizer_delivery_uncertainty_calibration.csv`
- `outputs/010_forecast_control/<artifact_namespace>/latest/optimizer_delivery_uncertainty_summary.csv`

Related references:
- [002_operating_direction_spec.md](../000_governance/002_operating_direction_spec.md)
- [current_operating_approach.md](current_operating_approach.md)
- [operational_hypotheses.md](operational_hypotheses.md)

## Intent

The contract is interval-based, not raw-second-based.

The optimizer-facing unit is the locked billing interval:
- cadence: `15 minutes`
- feed shape: one row per future interval in the current replayed control cycle
- selection policy: request the latest available stacked layer in priority
  order `nowcast -> phase -> hourly -> day_ahead`, then resolve to the
  freshest usable layer if the requested layer is stale or unavailable
- the priority order is structural; in the current latest Stage-10 bundle, the
  phase slot stays live because both `phase_stack_guard_policy` and
  `phase_rolling_support_policy` are `phase_candidate`

## Required Fields

Each delivery row should include:

| Field | Meaning |
|-------|---------|
| `cycle_origin_timestamp` | As-of timestamp for the replayed control cycle |
| `as_of_timestamp` | Explicit optimizer-facing alias for the cycle as-of time |
| `interval_start` | Target interval start |
| `interval_end` | Target interval end |
| `lead_interval_index` | 0-based interval offset from the cycle origin |
| `horizon_minutes` | Minutes from the cycle origin to the interval end |
| `requested_is_predicted_peak_interval` | Whether the requested layer marks the interval as the cycle peak |
| `operating_regime` | Coarse operating-context bucket used by the dynamic minute controller |
| `actual_ramp_band` | Interval ramp-context bucket derived from minute-level activity |
| `high_ramp_fraction` | Share of minute rows inside the interval flagged as high-ramp |
| `producer_stage` | Which stage emitted the delivery row |
| `contract_version` | Contract version for the emitted row shape |
| `run_id` | Persisted Stage-10 run id |
| `config_hash` | Stable config fingerprint for provenance |
| `requested_layer_role` | Latest layer available before freshness fallback |
| `requested_candidate_label` | Candidate label attached to the requested layer |
| `nowcast_dynamic_overlay_enabled` | Whether the dynamic minute controller has enough advisory support to run |
| `nowcast_dynamic_overlay_enforced` | Whether that controller is allowed to alter live layer selection |
| `nowcast_dynamic_overlay_eligible` | Whether the interval is strategically eligible for the learned minute overlay |
| `nowcast_dynamic_overlay_reason` | Why the controller kept or would demote the learned minute overlay |
| `selected_layer_role` | Which layer supplied the final interval forecast |
| `selected_candidate_label` | Persisted candidate label for provenance |
| `effective_forecast_as_of` | Timestamp of the attached forecast instance |
| `expected_layer_cadence_minutes` | Nominal update cadence for the selected layer |
| `forecast_age_minutes` | Age of the forecast at the emitted as-of time |
| `stale_threshold_minutes` | Operational freshness threshold for the selected layer |
| `is_stale_forecast` | Whether the selected forecast exceeded that threshold |
| `fallback_applied` | Whether the requested layer had to degrade to an older layer |
| `fallback_from_layer_role` | First layer that forced fallback |
| `fallback_to_layer_role` | Older layer chosen after fallback |
| `fallback_trigger` | Primary trigger such as `stale` or `unavailable` |
| `forecast_value` | Point forecast for the interval |
| `forecast_lower_80`, `forecast_upper_80` | Empirical 80% interval band |
| `forecast_lower_95`, `forecast_upper_95` | Empirical 95% interval band |
| `fallback_reason` | Why the selected layer was used |
| `resolution_path` | Compact trace of the layer-resolution chain |
| `quantile_source` | Whether the band came from lead-specific calibration or a layer-global fallback |
| `calibration_sample_n` | Effective support behind the emitted band |
| `confidence_score` | Heuristic 0-1 trust score derived from width, support, layer role, and freshness |
| `confidence_tier` | Digestible trust tier derived from `confidence_score` |

## Backtest-Only Fields

The preview CSV includes several backtest-only fields that are useful for
validation but would not be part of a live optimizer API by default:

| Field | Purpose |
|-------|---------|
| `actual_interval_mean` | Held-out truth for replay scoring |
| `selected_abs_error` | Backtest error of the selected forecast |
| `within_80_band`, `within_95_band` | Empirical coverage checks |
| `is_actual_peak_interval` | Whether the interval is the realized cycle peak |
| `is_predicted_peak_interval` | Whether the selected forecast marks it as the cycle peak |

## Uncertainty Semantics

The current uncertainty surface is intentionally modest and explicit.

What it is:
- empirical residual bands calibrated from held-out control-calibration windows
- layer-aware
- lead-aware when enough calibration support exists
- peak-aware when predicted-peak support exists
- next-lock-aware when lead-0 support is too sparse for a full lead-specific row
- explicitly downgraded to a layer-global fallback only when the more local
  contexts are too sparse

What it is not:
- a claim that the repo already has a best-in-class probabilistic model
- a substitute for full quantile-model training or conformal forecasting

This is an honest first risk surface, not a marketing claim.

## Fallback Semantics

`fallback_reason` is part of the contract because the optimizer should not be
forced to guess whether the best available forecast came from the full stack,
from a stale later update, or from a missing later update.

Current layer-priority logic:
- `nowcast`: full stacked forecast available
- `phase`: minute layer unavailable and a stack-approved phase correction exists;
  in the current latest bundle this slot resolves to the hourly candidate
  because the rolling-support guard vetoed a distinct phase correction
- `hourly`: phase and minute layers unavailable
- `day_ahead`: intraday updates unavailable

When wall-clock freshness is applied, the resolver can also emit:
- `nowcast_stale` and degrade to `phase`
- `phase_stale` and degrade to `hourly`
- `hourly_stale` and degrade to `day_ahead`

The machine-readable fallback and freshness policy now also lives in:
- `optimizer_operational_policy.json`

The dynamic minute controller now has its own persisted audit artifact too:
- `optimizer_dynamic_overlay_shadow_summary.json`
- `optimizer_dynamic_overlay_soft_summary.json`
- `optimizer_dynamic_overlay_soft_candidates.csv`

## Provenance

Every Stage-10 bundle also carries:
- `run_id`
- `config_hash`
- selected candidate labels per layer
- refresh policy
- phase stack guard policy
- phase rolling-support policy

Those live in `optimizer_delivery_contract.json`,
`optimizer_operational_policy.json`, and the broader `control_policy.json` /
`run_manifest.json` pair.

## Confidence And Freshness Semantics

The preview now carries two extra operational surfaces:

- freshness fields:
  `effective_forecast_as_of`, `forecast_age_minutes`,
  `stale_threshold_minutes`, `is_stale_forecast`
- confidence fields:
  `confidence_score`, `confidence_tier`

Important caveat:
- the standard replay preview is still emitted at the cycle origin, so
  `forecast_age_minutes` is usually `0`
- the serving preview keeps the same contract fields but drops replay-only truth
  columns
- the resolver itself is already executable inside Stage-10 and can be applied
  against a later `as_of_timestamp` for live-style fallback behavior
- `confidence_score` is intentionally a trust hint, not a probability

## Operational Policy Artifact

`optimizer_operational_policy.json` is the machine-readable companion to the
contract. It records:

- layer cadence and stale thresholds
- fallback order
- the current minute-layer policy
- the persisted dynamic-overlay shadow verdict
- day-ahead refresh thresholds and trigger mode
- runtime portability guidance for CPU-only, ARM64, and accelerated hosts
- retraining/review guidance

Related standalone minute-policy artifact:
- `outputs/005_performance/<artifact_namespace>/latest/operating_policy.json`

That Stage-5 artifact now makes one important distinction explicit:
- standalone `1m` can stay baseline-led
- Stage-10 can still use learned minute overlays as corrective specialists when
  the held-out control replay supports them

## Honest Current Limitations

The contract is now explicit, but the repo still has work left before anyone
should call it bankable best-in-class:
- the standalone `1m` anchor is still baseline-led
- uncertainty is empirical and replay-calibrated, not yet model-native
- Stage-10 now applies explicit next-lock, peak, and broader rolling-support
  guardrails to the phase slot; the latest persisted bundle keeps that slot on
  a distinct stack-aware phase correction, but that still needs faster replay
  and broader evidence before anyone should overclaim
- the dynamic minute controller exists, but the latest persisted shadow summary
  says hard enforcement would materially worsen all-interval error, so it
  remains shadow-only; the first soft-overlay shadow search also says pure
  nowcast still beats every admissible blended policy on the current replay
- live layer fallback now exists at the contract surface, but feature freshness,
  missing-input flags, and drift alarms still need a deeper serving/runtime
  implementation
