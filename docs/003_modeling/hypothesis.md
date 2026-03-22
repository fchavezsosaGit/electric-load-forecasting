# Historical Hypotheses

This document retains the original Report IV hypotheses that connected EDA
findings to the early notebook-first modeling surface in
`notebooks/003_modeling.ipynb`.

These hypotheses are kept for traceability. They explain why the repository
started at `1min`, why the early work emphasized notebook validation, and why
the project originally asked "can one learned model beat the baseline?"

They are no longer the repo's primary operating hypotheses.

Current decision-making has moved to:
- [operational_hypotheses.md](operational_hypotheses.md)
- [optimizer_delivery_contract.md](optimizer_delivery_contract.md)
- [current_operating_approach.md](current_operating_approach.md)
- [002_operating_direction_spec.md](../000_governance/002_operating_direction_spec.md)

Related references:
- [000_spec.md](../000_governance/000_spec.md)
- [mvmp.md](mvmp.md)
- [feature_sets.md](feature_sets.md)
- [glossary.md](../004_reference/glossary.md)

## Disposition Summary

| ID | Original name | Historical status | Current disposition | Why |
|----|---------------|-------------------|---------------------|-----|
| H1 | Workday signal | Evaluated | Retained as historical evidence | Workday effects remain important, but they are now part of the broader stacked feature surface rather than a standalone gate |
| H2 | Lag/rolling value | Evaluated | Retained as historical evidence | Transition-aware features still matter, but the real question is now stacked control value, not a notebook RMSE delta alone |
| H3 | Resolution tradeoff | Implemented beyond the notebook | Modified into multi-horizon operating policy | Resolution is no longer a `5min` vs `1min` binary; it is a horizon-specific selection problem |
| H4 | Nonlinear vs linear behavior | Evaluated | Retained as historical evidence | Model-family comparison still matters, but only inside objective-aware selection and stacked replay |
| H5 | Forecast horizon degradation | Implemented | Modified into the operational capability-envelope and delivery-contract story | There is no single crossover horizon; the repo now keeps a layered, objective-aware horizon envelope |

---

## Direction Evolution

The biggest change is that the repo no longer treats "prove a learned `1m`
anchor wins globally" as the main operating objective.

That direction was retired because the current evidence says:

- the canonical Stage-5 holdout is narrow and still baseline-led:
  [`../../outputs/005_performance/commercial_facility/latest/holdout_coverage_summary.json`](../../outputs/005_performance/commercial_facility/latest/holdout_coverage_summary.json)
  and
  [`../../outputs/005_performance/commercial_facility/latest/deployment_recommendation.json`](../../outputs/005_performance/commercial_facility/latest/deployment_recommendation.json)
- the broader advisory `1m` surface is learned-positive mainly in transition
  and high-ramp regimes:
  [`../../outputs/005_performance/commercial_facility/latest/supplemental_surface_advisory.json`](../../outputs/005_performance/commercial_facility/latest/supplemental_surface_advisory.json)
- the Stage-10 stack gets most of its value from the hourly layer and the final
  minute overlay, while the applied phase slot is currently hourly passthrough:
  [`../../outputs/010_forecast_control/commercial_facility/latest/control_backtest_summary.csv`](../../outputs/010_forecast_control/commercial_facility/latest/control_backtest_summary.csv)
  and
  [`../../outputs/010_forecast_control/commercial_facility/latest/rolling_control_layer_inference.csv`](../../outputs/010_forecast_control/commercial_facility/latest/rolling_control_layer_inference.csv)
- hard dynamic gating is not live-ready:
  [`../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_shadow_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_shadow_summary.json)

Research-aligned source credit for that shift now lives in
[002_operating_direction_spec.md](../000_governance/002_operating_direction_spec.md).

## Why These Stayed

The original hypotheses are not deleted because failure was informative:
- they explain why the project started at `1min`
- they show where the single-model story broke down
- they make it clear that the move toward a layered stack was evidence-driven,
  not scope drift

The rest of this file keeps the original H1-H5 definitions so older notebook
references and report links still make sense.

## H1: Workday signal

Observation:
EDA shows clear load-profile separation by business-day type (`full`, `half`, `none`).

Hypothesis:
We hypothesize that workday-aware signal reduces validation MAE by at least **10%**
relative to a temporal control without `workday`.

Experimental design:
- Resolution: `1min`
- Model family for primary readout: Ridge (`alpha` in `{0.1, 1.0, 10.0}`)
- Control: `temporal` feature set with `workday` removed in-notebook
  (`temporal_no_workday`)
- Treatment: `minimal` (`workday`, `hour`, `lag_1`)
- Evaluation split: validation (days 26-28)

Metric and target:
- Primary metric: MAE
- Target: >=10% MAE improvement

## H2: Lag and transition context

Observation:
EDA shows strong autocorrelation and transition behavior in `avg_load`, including
multi-horizon memory effects.

Hypothesis:
We hypothesize that lag/rolling enriched features reduce large transition errors and
improve validation RMSE by at least **8%** compared with temporal-only context.

Experimental design:
- Resolution: `1min`
- Primary comparison: `temporal` vs `curated`
- Models: Ridge and HistGradientBoostingRegressor cross-checks
- Evaluation split: validation (days 26-28)
- Feature surface: both sets now include continuous daily/weekly Fourier encoding;
  the comparison isolates richer lag/rolling/trend context rather than withholding
  cyclical structure from one side.
- Note: RMSE is primary for H2 because it emphasizes larger misses.
- Note: Ridge drops feature-NaN rows during fit, so `curated`/`full` can have fewer
  effective training rows than `minimal`/`temporal`; HGB is included as a cross-check
  because it can train with feature NaNs without that row-drop behavior.

Metric and target:
- Primary metric: RMSE
- Target: >=8% RMSE improvement

## H3: Resolution tradeoff (implemented outside the MVP notebook)

Observation:
Coarser resolutions can reduce noise and training cost, but may lose short-horizon
fidelity.

Hypothesis:
We hypothesize that a `5min` model can achieve MAE within **5%** of an equivalent
`1min` model while using fewer training rows.

Status:
- **Deferred inside the canonical Report IV notebook**, which remains `1min` by design.
- **Implemented in the repository runtime surface** through
  `scripts/005_multires_compare.py` and `run_pipeline.py --stage multires`.
- Current selection logic uses matched-horizon evaluation plus coverage/stability/runtime
  gates rather than relying on native-step MAE alone.
- Matched-horizon representability now uses exact second divisibility, so the
  configured second-level cadences (`1s`, `5s`, `10s`, `30s`) are valid research
  candidates instead of being restricted to minute-aligned comparisons.
- Matched-horizon learned candidates now evaluate both `recursive` and
  `direct_endpoint` strategies. The selection summary records which strategy actually
  cleared the gates.
- The latest Stage-6 evidence is mixed rather than uniformly persistence-led:
  - latest smoke run (`outputs/006_multires/commercial_facility/latest_smoke/`)
    - `15m`: learned winner at `30s/minimal/ridge-medium/direct_endpoint`
    - `60m`: learned winner at `1min/minimal/hgb-balanced/recursive`
  - `30m`: learned winner at `5min/curated/hgb-balanced/direct_endpoint`
  - `60m`: focused tuning run (`outputs/006_multires/commercial_facility/20260310T005916684602Z/`)
    selects `5min/minimal/hgb-balanced/recursive` with endpoint/path MAE
    `1148.166851` (`42.691558%`) / `1151.446627` (`36.611950%`)
  - `120m`: learned winner at `5min/curated/hgb-balanced/recursive`
- Research conclusion: learned multires value is evidenced across both smoke and
  targeted runs, but `60m` remains profile-sensitive because the broader candidate
  sweep can still keep persistence while smoke and focused tuning now both promote
  learned winners under the current model family and feature surface.

Implemented comparison design:
- Same feature-set/model grid across resolutions within one Stage-6 run.
- Native-step metrics are still emitted for diagnostics.
- Final selection uses matched-horizon recursive evaluation on a shared real-time horizon.
- Selection summary can explicitly keep persistence as the correct answer.

## H4: Nonlinear model behavior (exploratory)

Observation:
Feature interactions and regime nonlinearities may not be captured by purely linear
models.

Hypothesis (exploratory):
We expect nonlinear learners to show selective gains on some feature sets/hours, but
not necessarily a global MAE win over regularized linear baselines.

Experimental design:
- Resolution: `1min`
- Models:
  - Ridge: light/medium/strong regularization
  - HistGradientBoostingRegressor: conservative/balanced/aggressive
- Shared feature surface now includes continuous cyclical encoding (`hour_sin`,
  `hour_cos`, `dow_sin`, `dow_cos`) so nonlinear models can exploit smooth periodic
  structure without relying on hard hour/day bucket edges alone.
- Evaluation split: validation
- Additional holdout check: one-shot test after selection

Metric framing:
- Track both MAE and RMSE.
- No hard pass/fail threshold for H4; interpret as comparative behavior analysis.

## H5: Forecast horizon degradation

Observation:
The project goal is to deliver optimizer-ready load predictions. A downstream optimizer
needs to know which forecast lead times it can trust. Existing results already showed
strong horizon sensitivity: persistence is difficult to beat at `1m`, while the broader
rollout stack can outperform persistence at some longer horizons.

Hypothesis:
We hypothesize that a crossover horizon exists beyond which the selected model no longer
outperforms the persistence baseline, and that characterizing this degradation profile
provides the accuracy envelope a downstream optimizer requires.

Experimental design:
- `1m`: Stage-5 promoted holdout candidate vs persistence using the one-shot holdout
  artifact
- `15m` through `1440m`: Stage-7 challenger sweeps evaluated under
  `origin_policy=uniform`, with selection driven by `path_mae`
- Stage-8 (`scripts/008_horizon_curve.py`, canonical implementation:
  `scripts/modeling/horizon_curve.py`) consolidates those verified point results into
  a single horizon curve
- Output artifacts:
  - `outputs/009_horizon_curve/<artifact_namespace>/horizon_curve_summary.csv`
  - `outputs/009_horizon_curve/<artifact_namespace>/horizon_curve_candidates.csv`
  - `outputs/009_horizon_curve/<artifact_namespace>/crossover_summary.json`
  - `outputs/009_horizon_curve/<artifact_namespace>/fig_horizon_ratio_curve.png`
  - `outputs/009_horizon_curve/<artifact_namespace>/fig_horizon_absolute_mae.png`

Metric and target:
- Primary metric: MAE ratio (model MAE / persistence MAE) at each lead time
- Secondary readout: ratio to the strongest non-persistence baseline (`avg_workday`,
  `previous_day`, or other configured baselines)
- Target: characterize where the current learned stack is better or worse than
  persistence and the strongest baseline
- No hard pass/fail threshold; the deliverable is the degradation profile itself

Status:
- Stage-8 is executed and the full current degradation curve now exists under
  `outputs/009_horizon_curve/commercial_facility/latest/`.
- The result is not a single crossover point. It is a non-monotonic capability envelope
  because the strongest verified candidate changes by horizon.
- Current path-MAE outcome versus persistence:
  - wins at `15m`, `30m`, `60m`, `360m`, `720m`, `1440m`
  - losses at `1m`, `120m`, `240m`
- Current endpoint-MAE outcome versus persistence:
  - wins at `15m`, `30m`, `60m`, `240m`, `720m`, `1440m`
  - losses at `1m`, `120m`, `360m`
- Important operational nuance: the standalone Stage-5 `1m` holdout still keeps
  persistence as the deployment recommendation, but the Stage-10 stacked
  day-ahead/hourly/phase/nowcast replay now shows a strong learned `1m` nowcast
  layer on exact-control cycles. For optimizer-facing delivery, the current
  evidence supports using the Stage-10 stacked control surface rather than
  interpreting the Stage-5 `1m` holdout in isolation.
- The strongest current long-horizon result is still `1440m`:
  - learned candidate `10min/minimal/hgb-balanced`
  - learned endpoint/path MAE `968.909580` / `783.077104`
  - `avg_workday` endpoint/path MAE `986.676302` / `850.145715`
  - persistence endpoint/path MAE `1119.137272` / `1010.620668`
- The clearest current failure mode is `120m`, where the selected learned candidate
  loses badly on both endpoint and path MAE.
- `240m` remains mixed: the current learned candidate wins endpoint MAE but loses path
  MAE, so it is not yet rollout-stable.
- The next H5 model work is now horizon-specific rather than infrastructural:
  tune `120m` first, then recover `240m` path quality.
