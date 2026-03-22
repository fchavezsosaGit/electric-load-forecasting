# Operating Direction Implementation Plan

| Field | Value |
|---|---|
| Status | Active |
| Created | 2026-03-22 |
| Last Updated | 2026-03-22 |
| Scope | Execution plan for the current optimizer-facing forecasting direction |

This document translates the active direction in
[002_operating_direction_spec.md](002_operating_direction_spec.md) into an
implementation roadmap with acceptance criteria.

It is intentionally operational:

- what to build
- why it matters
- how we will judge it
- what must stay shadow-only until the evidence changes

## Current Starting Point

The repo already has a strong layered control stack, but it is not yet a
bankable "best-in-class" forecasting surface.

Current empirical anchors:

- Standalone Stage-5 `1m` remains baseline-led on the canonical gate.
  [`deployment_recommendation.json`](../../outputs/005_performance/commercial_facility/latest/deployment_recommendation.json)
- The Stage-5 broader advisory surface is learned-positive in transition-heavy
  regimes and on high-ramp behavior.
  [`supplemental_surface_advisory.json`](../../outputs/005_performance/commercial_facility/latest/supplemental_surface_advisory.json)
- The Stage-10 stacked system is strong on the optimizer-facing replay surface.
  [`control_backtest_summary.csv`](../../outputs/010_forecast_control/commercial_facility/latest/control_backtest_summary.csv)
- The active phase slot is structural, but not entitled. It currently resolves
  to hourly passthrough when it cannot justify itself.
  [`control_policy.json`](../../outputs/010_forecast_control/commercial_facility/latest/control_policy.json)
- Hard dynamic minute gating is harmful and must remain shadow-only.
  [`optimizer_dynamic_overlay_shadow_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_shadow_summary.json)
- Uncertainty is honest but still too wide in the most optimizer-sensitive
  slices.
  [`optimizer_delivery_uncertainty_summary.csv`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_delivery_uncertainty_summary.csv)
- Cold Stage-10 runtime is still heavier than it should be.
  [`runtime_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/runtime_summary.json)

## Research Basis

The plan below is informed by, but not subordinate to, the literature.
Promotion still depends on repo artifacts first.

- [Athanasopoulos et al., "Forecasting with Temporal Hierarchies"](https://robjhyndman.com/publications/temporal-hierarchies/)
- [Rangapuram et al., "Coherent Probabilistic Forecasting of Temporal Hierarchies"](https://proceedings.mlr.press/v206/rangapuram23a.html)
- [Elmachtoub and Grigas, "Smart Predict, then Optimize"](https://arxiv.org/abs/1710.08005)
- [Bates and Granger, "The Combination of Forecasts"](https://www.tandfonline.com/doi/abs/10.1057/jors.1969.103)
- [Xu and Xie, "Conformal Prediction Interval for Dynamic Time-Series"](https://proceedings.mlr.press/v139/xu21h.html)
- [Gibbs and Candès, "Adaptive Conformal Inference Under Distribution Shift"](https://arxiv.org/abs/2106.00170)
- [MAPIE TimeSeriesRegressor documentation](https://mapie.readthedocs.io/en/stable/generated/mapie.regression.TimeSeriesRegressor.html)
- [scikit-learn HistGradientBoostingRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [XGBoost quantile regression example](https://xgboost.readthedocs.io/en/stable/python/examples/quantile_regression.html)

## Guiding Rules

1. Keep the architecture layered.
   Day-ahead, hourly, structural `15m`, and minute-correction layers solve
   different problems and should not be forced into one winner-take-all model.
2. Keep standalone `1m` honest.
   Supplemental evidence can inform Stage-10 policy, but it must not override
   the canonical Stage-5 deployment gate.
3. Prefer soft adaptation over brittle hard gates.
   When minute intelligence helps only in some regimes, trust-weighted blending
   is preferred over abrupt fallback unless the hard gate becomes clearly
   positive on replay.
4. Treat uncertainty and fallback behavior as model quality.
   A narrow but dishonest interval is worse than a wider truthful one.
5. Promotion must be optimizer-aware.
   `next_lock_mae`, peak metrics, and stacked replay matter more than generic
   single-surface leaderboard wins.

## Phased Plan

### Phase 1: Regime-Aware Soft Minute Overlay

Status: initial shadow-search slice implemented; still shadow-only

Why:

- Hard gating from nowcast to phase/hourly is too destructive because it drops
  too many background rows to a weaker layer.
- The Stage-5 supplemental surface says learned minute behavior helps in
  specific operating regimes, not necessarily everywhere.

What to implement:

- Add a soft minute-overlay controller in Stage-10 shadow analysis.
- Blend between the upstream interval forecast and the learned minute overlay
  instead of using only binary on/off routing.
- Keep strategic intervals pinned to full nowcast weight:
  next lock, predicted peak, and any future must-hold slices justified by
  evidence.
- Keep the controller diagnostic-only until it proves itself.
- Persist a dedicated artifact that records:
  candidate soft-weight policies,
  selected best policy under replay,
  whether it beats pure nowcast,
  whether it stays non-regressive on next-lock and peak-hit behavior.

Acceptance criteria:

- A new Stage-10 artifact persists the soft-overlay evaluation and final
  recommendation.
- Soft-overlay configuration is centralized in TOML and `config.py`.
- The best soft policy is selected by replay, not hard-coded.
- If no policy beats pure nowcast without optimizer-facing regressions, the
  artifact must say so explicitly and recommend staying with pure nowcast.
- Tests cover background intervals, regime-supported intervals, strategic
  intervals, and policy recommendation behavior.

Latest result:

- The first shadow-search implementation is now live in Stage-10 artifacts.
- The current replay evaluated `29` soft-overlay policies and still chose pure
  nowcast (`soft_overlay_sw100_bw100`) as the best admissible policy.
- That is a useful outcome, not a failure: it means the repo now has direct
  replay evidence that hard gating is harmful and background softening is not
  yet improving the active minute policy either.

Do not do:

- Do not make the soft controller live by default in the same change.
- Do not let the soft controller override standalone Stage-5 deployment policy.

### Phase 2: Conformal Uncertainty Tightening

Status: next after Phase 1

Why:

- Current intervals are honest, but next-lock and peak-conditioned widths are
  still wider than we want.
- The repo needs calibrated operational trust, not only retrospective summary
  coverage.

What to implement:

- Add conformal-style calibration on top of the existing layer outputs.
- Start with the already useful optimizer-facing horizons:
  `15m`, `60m`, and `24h`.
- Keep `1m` uncertainty improvements secondary until the stronger horizons are
  sharper and better supported.
- Compare current heuristic residual bands against conformal alternatives.
- Persist both coverage and sharpness diagnostics by scope:
  all intervals, next lock, actual peak, predicted peak, and layer role.

Acceptance criteria:

- A new artifact or expanded summary shows interval coverage and width for the
  incumbent versus the conformal challenger.
- Coverage remains at or above the configured target without silently widening
  all bands.
- Next-lock and peak-conditioned intervals show either narrower bands at equal
  coverage or better coverage at similar width.
- The serving preview and contract continue to emit the same core fields.

### Phase 3: Direct Quantile Forecasting Where It Matters

Status: gated on Phase 2 results

Why:

- Conformal calibration is model-agnostic and honest, but direct quantile
  models may better express asymmetric risk where enough signal exists.

What to implement:

- Add direct quantile challengers only for the horizons/layers with enough
  support and clear business value.
- Prioritize `15m`, `60m`, and `24h` over standalone `1m`.
- Compare direct quantile surfaces against the calibrated point-forecast bands.

Acceptance criteria:

- Quantile challengers are optional and host-safe.
- They enter evaluation only where config explicitly enables them.
- They beat or materially complement the incumbent uncertainty surface on at
  least one optimizer-relevant scope without degrading maintainability.
- If they do not help, the repo keeps the conformalized point-forecast path as
  the active direction and records that honestly.

### Phase 4: Runtime Reduction And Structural Cleanup

Status: active in parallel

Why:

- Cold Stage-10 replay is still too expensive.
- Runtime waste is now more likely to come from replay structure than from
  missing hardware utilization.

What to implement:

- Keep shrinking replay surfaces that are not changing policy.
- Reduce repeated work in phase and rolling replay.
- Continue demoting expensive phase logic when the active policy is passthrough.
- Split oversized control logic into clearer ownership boundaries only where it
  reduces risk and test burden.

Acceptance criteria:

- Cold Stage-10 wall clock improves meaningfully on a like-for-like run.
- Runtime summaries show a reduced replay share or smaller hotspot totals.
- Policy quality remains stable or improves.
- Config stays centralized and tests still cover new decision paths.

### Phase 5: Local Adaptive Residual Challenger

Status: later, shadow-only

Why:

- State-space or locally adaptive residual logic may still help online
  correction, but the evidence does not justify a global rewrite.

What to implement:

- Add a local residual-correction challenger as a shadow experiment only.
- Use it to adapt blend weights or residual correction, not to replace the
  layered stack.

Acceptance criteria:

- It remains optional and shadow-only unless replay proves clear value.
- It is evaluated against the same Stage-10 optimizer-facing metrics.
- It does not degrade the repo's cross-host portability or maintainability.

## Cross-Phase Evidence Gates

Every promotion decision should check all of these before the repo narrative
changes:

- Stage-5 deployment truth:
  [`deployment_recommendation.json`](../../outputs/005_performance/commercial_facility/latest/deployment_recommendation.json)
- Stage-5 regime support:
  [`holdout_coverage_summary.json`](../../outputs/005_performance/commercial_facility/latest/holdout_coverage_summary.json)
  and
  [`supplemental_surface_advisory.json`](../../outputs/005_performance/commercial_facility/latest/supplemental_surface_advisory.json)
- Stage-10 active policy:
  [`control_policy.json`](../../outputs/010_forecast_control/commercial_facility/latest/control_policy.json)
- Stage-10 stacked quality:
  [`control_backtest_summary.csv`](../../outputs/010_forecast_control/commercial_facility/latest/control_backtest_summary.csv)
- Stage-10 rolling support:
  [`rolling_control_layer_inference.csv`](../../outputs/010_forecast_control/commercial_facility/latest/rolling_control_layer_inference.csv)
- Stage-10 runtime:
  [`runtime_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/runtime_summary.json)
- Dynamic overlay truth:
  [`optimizer_dynamic_overlay_shadow_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_shadow_summary.json)
- Delivery uncertainty:
  [`optimizer_delivery_uncertainty_summary.csv`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_delivery_uncertainty_summary.csv)

## Risks And Watchouts

- The Stage-5 canonical holdout is still narrow. Supplemental regime evidence is
  useful, but it is not a license to overclaim.
- Soft overlay tuning can overfit a tiny replay surface if it is not guarded by
  simple grids, transparent artifacts, and non-regression checks.
- Conformal methods can preserve coverage by becoming too wide. Sharpness must
  be measured alongside coverage.
- Quantile models can increase complexity quickly. They should be added only
  where they buy real optimizer-facing value.
- Runtime optimization can hide correctness regressions if replay shortcuts are
  not covered by tests.

## Definition Of "Proceed"

The repo is ready to move from planning into the next round of work when all of
these are true:

- the active direction is documented in the current spec
- the execution plan and acceptance criteria are written down
- the next phase has an artifact target
- the config surface is clear enough to implement without scattering behavior
- the tests identify what would count as a regression

That bar is now met. The first active implementation slice is Phase 1:
regime-aware soft minute overlay in shadow mode.
