# Operational Hypotheses

This document defines the repo's current optimizer-facing hypotheses.

It does not replace the historical notebook hypotheses in
[hypothesis.md](hypothesis.md). Those older hypotheses explain why the project
started at `1min` and how the early modeling surface was formed. This document
replaces them only for current decision-making.

The current question is no longer "what is the single best model?" It is:

Can a layered forecast-plus-nowcast stack deliver an honest, optimizer-ready
interval feed with useful next-lock quality, peak awareness, explicit fallback
behavior, and uncertainty signals?

Related references:
- [002_operating_direction_spec.md](../000_governance/002_operating_direction_spec.md)
- [current_validation_snapshot.md](current_validation_snapshot.md)
- [current_operating_approach.md](current_operating_approach.md)
- [optimizer_delivery_contract.md](optimizer_delivery_contract.md)
- [model_and_blend_guide.md](model_and_blend_guide.md)

## Current Goal

The active goal is not to prove a single learned `1m` model wins everywhere.

The active goal is to deliver an honest, optimizer-ready layered interval feed
for this facility, with the right winner allowed to differ by layer when the
artifacts say it should.

Research and direction credit:

- [Athanasopoulos et al., "Forecasting with Temporal Hierarchies"](https://robjhyndman.com/publications/temporal-hierarchies/)
- [Rangapuram et al., "Coherent Probabilistic Forecasting of Temporal Hierarchies"](https://proceedings.mlr.press/v206/rangapuram23a.html)
- [Elmachtoub and Grigas, "Smart Predict, then Optimize"](https://arxiv.org/abs/1710.08005)
- [Xu and Xie, "Conformal Prediction Interval for Dynamic Time-Series"](https://proceedings.mlr.press/v139/xu21h.html)
- official [MAPIE time-series API documentation](https://mapie.readthedocs.io/en/stable/generated/mapie.regression.TimeSeriesRegressor.html)

These sources informed the repo's direction change, but the repo still follows
the persisted artifacts when research intuition and local evidence diverge.

## Summary Table

| ID | Name | Primary readout | Target | Current status |
|----|------|------------------|--------|----------------|
| O1 | Layered control value | `next_lock_mae`, `lock_mae`, peak metrics | Intraday layers must beat the frozen day-ahead path on held-out control replay | Supported |
| O2 | Minute layer as corrective overlay | stacked `next_lock_mae` and lock gain vs phase layer | `1m` may stay baseline-led in isolation, but the minute layer must help inside the full stack | Supported |
| O3 | Uncertainty is part of the product | empirical interval-band coverage and width | Delivery artifacts must emit usable interval bands and coverage evidence | Implemented, still calibrating |
| O4 | Delivery contract is first-class | contract completeness and provenance | Every delivered interval row must include timestamps, horizon, forecast, fallback, confidence fields, and model provenance | Implemented |
| O5 | Operational robustness beats hidden cleverness | explicit refresh/fallback/drift behavior | The stack must stay honest under missing signal, hardware differences, and replay selection rules | Implemented, still hardening |
| O6 | Dynamic overlay earns live promotion | shadow vs enforced counterfactual on live-facing metrics | The minute controller stays shadow-only until enforcement no longer hurts the control surface | Shadow-only, not yet supported |

---

## O1: Layered control value

Observation:
The strongest empirical signal in the repo is not a standalone `1m` model.
It is the layered Stage-10 replay, where frozen day-ahead, hourly, phase, and
minute updates are evaluated together on shared control cycles.

Hypothesis:
We hypothesize that a layered stack will improve optimizer-relevant interval
quality over the frozen `24h` anchor, especially on:
- `next_lock_mae`
- `lock_mae`
- peak timing and peak-value error

Readouts:
- exact-control replay in `outputs/010_forecast_control/<artifact_namespace>/latest/`
- rolling benchmark replay in the same folder

Promotion rule:
- a later layer should only be trusted if it improves the stacked outcome on the
  held-out control surface
- isolated offline wins are not enough

Current interpretation:
- supported
- this is the repo's primary operational hypothesis

## O2: Minute layer as corrective overlay

Observation:
Stage-5 still honestly keeps `persistence` as the standalone `1m` deployment
recommendation, while Stage-10 shows a strong learned minute-layer correction
inside the full stack.

Hypothesis:
We hypothesize that the minute layer should be treated as a corrective overlay,
not as a mandatory standalone anchor. The right question is whether it improves
the stacked control path versus the already-corrected hourly-plus-phase path.

Success condition:
- positive stacked gain on held-out exact-control replay
- positive stacked gain on the broader rolling replay
- no claim that the standalone `1m` learned model is universally superior unless
  Stage-5 later proves it

Current interpretation:
- supported
- the repo should stay baseline-led at standalone `1m` when the holdout says so

## O3: Uncertainty is part of the product

Observation:
Point forecasts alone are not enough for optimizer consumption. The optimizer
needs a risk signal or interval band, especially near locked intervals and
potential peaks.

Hypothesis:
We hypothesize that empirical residual bands calibrated on held-out control
windows can provide a useful first uncertainty surface for the delivered
interval forecast.

Current implementation surface:
- `optimizer_delivery_uncertainty_calibration.csv`
- `optimizer_delivery_uncertainty_summary.csv`
- `optimizer_delivery_preview.csv`

Success condition:
- interval bands are emitted with explicit provenance
- empirical coverage is tracked on held-out replay
- lead-specific calibration is preferred, with explicit fallback to layer-global
  bands when support is too sparse

Current interpretation:
- implemented, but still in the "honest risk signal" phase rather than a final
  probabilistic-model claim

## O4: Delivery contract is first-class

Observation:
The project goal is pre-optimizer delivery. That means the interface contract is
part of model quality, not an afterthought.

Hypothesis:
We hypothesize that the repo is only truly useful when the forecast surface is
emitted as a stable interval contract with:
- `as_of` timestamp
- target interval start/end
- horizon
- selected layer
- selected candidate label
- point forecast
- interval bands
- fallback reason
- calibration support and provenance

Current implementation surface:
- `optimizer_delivery_contract.json`
- `optimizer_operational_policy.json`
- `optimizer_delivery_preview.csv`

Current interpretation:
- implemented
- this is now part of the Stage-10 output contract

## O5: Operational robustness beats hidden cleverness

Observation:
Forecast quality is not enough if the stack silently depends on one machine,
one runtime, or one clean data regime.

Hypothesis:
We hypothesize that the repo will be more trustworthy if it remains explicit
about:
- which layer is selected
- when refresh is triggered
- what fallback path was used
- when optional GPU acceleration is present versus absent
- when calibration support is lead-specific versus global fallback
- when a selected forecast should be treated as stale
- how a downstream consumer should interpret the confidence signal

Success condition:
- ARM64 and non-accelerated teammates stay on safe CPU paths
- accelerated hosts can use optional GPU candidates where they actually help
- policy artifacts explain why the repo selected the path it did
- the delivery preview rows expose freshness and trust metadata directly

Current interpretation:
- implemented, still hardening
- the contract and policy surface are now explicit, but live stale-signal
  enforcement and retraining automation still need deeper operational work

## O6: Dynamic overlay earns live promotion

Observation:
The repo now has a dynamic minute-overlay controller, but the latest persisted
shadow analysis shows that hard enforcement would worsen all-interval selected
absolute error even though it leaves next-lock and peak-hit unchanged.

Hypothesis:
We hypothesize that the dynamic overlay should stay shadow-only until its
persisted counterfactual can improve or at least not regress live-facing Stage-10
metrics.

Current implementation surface:
- `optimizer_dynamic_overlay_shadow_summary.json`
- `optimizer_delivery_preview.csv`
- `optimizer_delivery_serving_preview.csv`

Success condition:
- shadow analysis stays persisted in the current Stage-10 bundle
- live enforcement remains disabled until the counterfactual is non-harmful on
  all-interval error and does not regress next-lock or peak behavior
- future promotion should prefer soft weighting or trust weighting over hard
  gating when the shadow evidence remains mixed

Current interpretation:
- shadow-only, not yet supported for live enforcement
