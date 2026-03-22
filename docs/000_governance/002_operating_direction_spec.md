# Operating Direction Specification

| Field | Value |
|---|---|
| Status | Active |
| Created | 2026-03-22 |
| Last Updated | 2026-03-22 |
| Scope | Current optimizer-facing modeling and delivery direction |

This document is the active implementation-direction spec for the repo's
forecasting and nowcasting work.

It does not replace the foundation hardening specs in
[000_spec.md](000_spec.md) and [001_spec.md](001_spec.md). Those remain true for
pipeline and notebook infrastructure. This document replaces the older
single-model and strict `1min`-first planning story for current operating
decisions.

The executable follow-on plan lives in
[003_operating_direction_implementation_plan.md](003_operating_direction_implementation_plan.md).

## Current Goal

Deliver an honest, optimizer-ready pre-optimizer interval feed for one
commercial facility by combining:

- a strong frozen day-ahead anchor
- an hourly corrective layer
- a structural `15m` slot that only stays live when stack evidence supports it
- a minute overlay that is allowed to be learned only when replay evidence
  proves it helps
- uncertainty, freshness, provenance, and fallback metadata that let a
  downstream optimizer decide how hard to trust each interval

The goal is not to prove one universal model wins everywhere.

## Why The Direction Changed

The repo started with a notebook-era question:

"Can a learned `1m` model beat persistence and become the anchor?"

The latest evidence says that framing was too narrow.

Empirical reasons:

- Stage-5 canonical holdout remains narrow and baseline-led.
  [`holdout_coverage_summary.json`](../../outputs/005_performance/commercial_facility/latest/holdout_coverage_summary.json)
  shows only `3` days and one observed operating regime
  (`none_inactive`), and
  [`deployment_recommendation.json`](../../outputs/005_performance/commercial_facility/latest/deployment_recommendation.json)
  still recommends `persistence`.
- Broader leakage-safe Stage-5 advisory evidence is useful, but it is not the
  deployment gate. [`supplemental_surface_advisory.json`](../../outputs/005_performance/commercial_facility/latest/supplemental_surface_advisory.json)
  shows the learned `1m` challenger beating persistence on `8614` rows across
  `6` days, with support concentrated in `transition_only` and
  `transition_active`.
- Stage-10 exact and rolling replay show the stack is where the value is.
  [`control_backtest_summary.csv`](../../outputs/010_forecast_control/commercial_facility/latest/control_backtest_summary.csv)
  and
  [`rolling_control_layer_inference.csv`](../../outputs/010_forecast_control/commercial_facility/latest/rolling_control_layer_inference.csv)
  show the hourly layer and minute overlay drive the gains, while the currently
  applied phase slot resolves to hourly passthrough.
- Hard dynamic gating is not live-ready.
  [`optimizer_dynamic_overlay_shadow_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_shadow_summary.json)
  shows that enforcing the current gate would raise all-interval selected
  absolute error from `47.503499` to `417.163949`, so the dynamic controller
  must stay shadow-only.
- Runtime is still a first-class constraint.
  [`runtime_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/runtime_summary.json)
  shows the latest cold Stage-10 bundle still spending most of its time in
  replay, especially `replay_phase_layer`.

## Research Credit

These sources influenced the repo's shift away from a single-model story and
toward a layered, decision-aware contract.

- [Athanasopoulos et al., "Forecasting with Temporal Hierarchies"](https://robjhyndman.com/publications/temporal-hierarchies/)
  helped frame the problem as horizon-specific and coherence-sensitive rather
  than a single-resolution winner-take-all search.
- [Rangapuram et al., "Coherent Probabilistic Forecasting of Temporal Hierarchies"](https://proceedings.mlr.press/v206/rangapuram23a.html)
  reinforced that forecast quality and uncertainty should stay coherent across
  horizons, which matches our day-ahead plus intraday stack.
- [Elmachtoub and Grigas, "Smart Predict, then Optimize"](https://arxiv.org/abs/1710.08005)
  pushed the repo toward next-lock, peak, and optimizer-facing selection logic
  instead of generic leaderboard MAE alone.
- [Xu and Xie, "Conformal Prediction Interval for Dynamic Time-Series"](https://proceedings.mlr.press/v139/xu21h.html)
  and the official [MAPIE time-series documentation](https://mapie.readthedocs.io/en/stable/generated/mapie.regression.TimeSeriesRegressor.html)
  informed the uncertainty direction: honest, adaptive intervals should sit in
  the product surface, not only in offline evaluation tables.

These papers inspired the direction. They were not copied mechanically, and the
repo still follows the artifact-backed evidence above when those ideas conflict
with observed behavior.

## Active Operating Thesis

The current thesis is:

1. Use a layered stack because different horizons solve different problems.
2. Keep the canonical Stage-5 `1m` holdout gate honest, even when the broader
   advisory surface is learned-positive.
3. Let Stage-10 promote learned minute overlays only when the held-out control
   replay supports them.
4. Treat the phase slot as structural but not entitled. If it does not improve
   the stacked control path, it should resolve to hourly passthrough.
5. Treat uncertainty, freshness, fallback, and provenance as part of model
   quality, not delivery afterthoughts.

## Retired Or Deprioritized Directions

These are explicitly not the active path right now:

- No blanket "best-in-class" or "state of the art" claim while Stage-5 still
  keeps `persistence` as the standalone `1m` recommendation.
- No single global best-model narrative across `1m`, `15m`, `60m`, and `24h`.
- No full Kalman or global latent-state rewrite as the main architecture.
- No hard live enforcement of the current dynamic minute gate.
- No sub-minute default promotion without new gate-clearing evidence.
- No API-first work ahead of model, uncertainty, and replay quality.
- No distinct live phase correction until it re-earns promotion on the exact and
  rolling control surfaces.

## Implementation Phases

### Phase A: Truth And Policy Wiring

Status: implemented

What this phase means:

- current-state docs derive from persisted artifacts
- Stage-5 writes a minute operating policy
- Stage-10 writes a delivery contract, fallback policy, uncertainty artifacts,
  and dynamic-overlay shadow analysis
- promotion logic is optimizer-aware rather than generic-leaderboard-only

Primary evidence:

- [`current_validation_snapshot.md`](../003_modeling/current_validation_snapshot.md)
- [`current_operating_approach.md`](../003_modeling/current_operating_approach.md)
- [`optimizer_delivery_contract.md`](../003_modeling/optimizer_delivery_contract.md)

### Phase B: Runtime Reduction And Structural Discipline

Status: in progress

What this phase means:

- keep reducing cold Stage-10 replay cost
- keep candidate pools small and evidence-backed
- do not spend runtime on a phase layer that is not earning its keep

Current watch item:

- [`runtime_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/runtime_summary.json)
  still shows replay-dominant cost, with `replay_phase_layer` the main hotspot

### Phase C: Regime-Aware Minute Overlay

Status: active next direction

What this phase means:

- keep standalone `1m` baseline-led where the canonical holdout says so
- use learned minute overlays as corrective specialists where broader evidence
  supports them
- prefer soft weighting or trust-weighting over hard gating until the shadow
  counterfactual becomes positive

Primary evidence:

- [`operating_policy.json`](../../outputs/005_performance/commercial_facility/latest/operating_policy.json)
- [`supplemental_surface_advisory.json`](../../outputs/005_performance/commercial_facility/latest/supplemental_surface_advisory.json)
- [`optimizer_dynamic_overlay_shadow_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_shadow_summary.json)

### Phase D: Uncertainty Sharpening

Status: active next direction

What this phase means:

- keep uncertainty honest
- tighten next-lock and peak-conditioned bands without hiding coverage loss
- favor conditional calibration and conformal-style methods over ad hoc optimism

Primary watch artifact:

- [`optimizer_delivery_uncertainty_summary.csv`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_delivery_uncertainty_summary.csv)

## Guardrails

- Do not let supplemental Stage-5 evidence override the canonical holdout
  deployment gate.
- Do not promote a phase or minute policy only because it wins an isolated
  benchmark; it must help the stacked control path.
- Do not call a dynamic controller "ready" until the persisted shadow summary is
  positive on live-facing metrics.
- Do not default to sub-minute cadences because hardware can handle them.
- Do not widen candidate pools unless the extra pool materially changes policy
  quality.
- Keep CPU-safe behavior correct on teammate ARM64 and non-accelerated hosts.

## What To Check Before Any New Claim

- Stage-5 deployment gate:
  [`deployment_recommendation.json`](../../outputs/005_performance/commercial_facility/latest/deployment_recommendation.json)
- Stage-5 regime support:
  [`holdout_coverage_summary.json`](../../outputs/005_performance/commercial_facility/latest/holdout_coverage_summary.json)
  and
  [`supplemental_surface_advisory.json`](../../outputs/005_performance/commercial_facility/latest/supplemental_surface_advisory.json)
- Stage-10 current policy:
  [`control_policy.json`](../../outputs/010_forecast_control/commercial_facility/latest/control_policy.json)
- Stage-10 stacked quality:
  [`control_backtest_summary.csv`](../../outputs/010_forecast_control/commercial_facility/latest/control_backtest_summary.csv)
- Stage-10 rolling support:
  [`rolling_control_layer_inference.csv`](../../outputs/010_forecast_control/commercial_facility/latest/rolling_control_layer_inference.csv)
- Stage-10 runtime:
  [`runtime_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/runtime_summary.json)
- Dynamic live-readiness:
  [`optimizer_dynamic_overlay_shadow_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_shadow_summary.json)

## Simple Decision Summary

If we proceed from here, the repo should keep doing this:

- preserve the layered day-ahead -> hourly -> structural `15m` -> minute design
- keep the phase slot structural but allow hourly passthrough to win
- keep standalone `1m` honest and baseline-led until canonical evidence changes
- use learned minute overlays where stacked replay supports them
- keep dynamic routing in shadow until the counterfactual turns positive
- spend the next quality effort on runtime, uncertainty, and regime-aware minute
  policy, not on a broad new-model detour
