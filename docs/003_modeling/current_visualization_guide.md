# Current Visualization Guide

This guide is generated from the latest persisted artifacts. It exists to
answer three recurring questions quickly:

- What are the repo's current goals and success criteria?
- Which metrics are authoritative for each decision layer?
- What does each visual answer, and what should you look for?

- Generated at: `2026-03-22T05:10:19.220542+00:00`
- Artifact namespace: `commercial_facility`
- Integrated dashboard: [../../outputs/reports/commercial_facility/latest/validation_dashboard.html](../../outputs/reports/commercial_facility/latest/validation_dashboard.html)

## Executive Summary

- 1-minute decision: `persistence`. Best learned gap vs persistence: 1.331 The current holdout covers a narrow operating regime. Broader advisory evidence is learned-positive in: transition_active, transition_only. Strongest broader segment: actual_ramp_band=high_ramp (0.878 ratio).
- Horizon wins: `8/9`. Current Stage-8 rows beating persistence on their selected objective.
- Control-layer lift: `93.810%`. Exact lock MAE reduction; rolling lock MAE reduction is 93.782%.
- Validation surface: `8/8 green`. Latest modeling, rollout, control, and notebook manifests available to the dashboard. Current Stage-10 hotspot: select_phase_stack_policy (174.917s).

## Goals

| Goal | Decision layer | Primary metric | Current readout | Status |
| --- | --- | --- | --- | --- |
| Honest 1-minute deployability | Stage-5 holdout gate | Holdout MAE versus persistence | Best learned curated_ramp/xgb-balanced/residual+blend at 175.055 vs persistence 173.724 (gap 1.331). | Not yet met |
| Objective-aware horizon coverage | Stage-8 horizon curve | Ratio to persistence on the selected horizon objective | Learned wins currently appear at 15m on next_lock_mae, 30m on next_lock_mae, 60m on next_lock_mae, 120m on next_lock_mae, 240m on profile_shape_mae, 360m on profile_shape_mae, 720m on profile_shape_mae, 1440m on profile_shape_mae. | Strong signal |
| Layered control-stack improvement | Stage-10 forecast control | Lock MAE and profile-shape MAE through the stacked layers | Exact lock error falls 93.810% and rolling lock error falls 93.782% from frozen day-ahead to the final nowcast layer. Phase gain is statistically supported on rolling evaluation. | Operationally strong |
| Reproducible evidence chain | End-to-end validation surface | Latest artifact and notebook manifest health | 8/8 latest manifests report success. | Green |

## Decision Layers

| Stage | Decision question | Authoritative metric | Main visual |
| --- | --- | --- | --- |
| Stage-4 notebook benchmark | What does the core 1-minute modeling surface look like before promotion or control replay? | MAE, RMSE, error-by-hour structure | Existing Stage-4 PNG gallery |
| Stage-5 holdout gate | Should a learned 1-minute model replace persistence on holdout right now? | Holdout MAE plus bootstrap support versus persistence | 1-minute Holdout Leaderboard |
| Stage-6 multiresolution | Which matched-horizon candidates earn their runtime cost when compared to persistence? | MAE ratio to persistence, coverage, runtime | Matched-Horizon Runtime vs Persistence |
| Stage-7 rollout selection | Which rollout policy wins once the objective changes from endpoint error to path, phase, or profile quality? | Selection-target metric by objective | Stage-7 winner table |
| Stage-8 horizon curve | At which horizons does the learned stack beat persistence on the right metric? | Objective-matched ratio to persistence | Objective-Aware Horizon Ratio Curve + Horizon Win Matrix |
| Stage-10 forecast control | Do layered updates reduce lock error and profile error on exact and rolling control cycles? | lock_mae, profile_shape_mae, rolling gain confidence | Control-Layer Trajectory, Cycle Distribution, and Refresh Policy Comparison |

## Success Metrics

| Metric | Use this when | Authoritative scope | A good pattern looks like | Do not over-interpret |
| --- | --- | --- | --- | --- |
| MAE | Native-unit average error for one facility. | Stage-4 and Stage-5 holdout decisions. | Lower than persistence on holdout and stable across folds. | Can hide whether the error is concentrated in expensive transition windows. |
| MAE% | Scale-normalized comparison across horizons and runs. | Cross-horizon reporting and cross-resolution comparisons. | Moves down with raw MAE when comparing across different load scales. | Do not treat it as a substitute for native-unit operating budgets. |
| RMSE | Penalty on large misses and transition spikes. | Stage-4 and Stage-5 transition-risk checks. | Improves with MAE when the model is not trading off mean error for larger outliers. | A lower RMSE alone does not justify deployment if MAE still loses to persistence. |
| MAE ratio to persistence | Simple parity test against the operational baseline. | Stage-6 and Stage-8 cross-horizon screening. | Below 1.0. | Only compare the ratio on the metric that matters for the decision layer. |
| next_lock_mae | Error on the next locked interval after an update. | Short corrective horizons such as 15m and 60m. | Large reductions versus persistence and strong phase/hourly corrections. | A candidate can win next-lock MAE while still miss whole-day profile quality. |
| profile_shape_mae | Shape quality after rescaling predicted energy to actual energy. | Day-ahead profile selection and refresh policy decisions. | Lower than persistence when total-energy bias is removed. | Do not use it alone when the business cost is tied to the next locked interval. |
| lock_mae | Operationally costly locked-interval error in the control backtest. | Stage-10 control-layer evaluation. | Each later layer reduces it materially and the gain remains visible on rolling evaluation. | Exact-window wins are not enough if rolling confidence intervals are flat. |
| energy_mae | Total energy mismatch over the control horizon. | Stage-10 whole-profile sanity checks. | Falls alongside profile-shape MAE for profile-oriented improvements. | A low energy miss can still hide poor minute-level timing. |

## Primary Markdown Embeds

Treat `Core inline` visuals as the default embeds for notebook narrative cells, canonical snapshots, and report summaries. Use `Context inline` visuals only when the section is specifically about compute tradeoffs or rollout realism, and reserve `Policy inline` visuals for control-policy and optimizer-facing markdowns.

| Stage | Visual | Embed tier | Primary markdown homes | Decision question |
| --- | --- | --- | --- | --- |
| Stage-4 Modeling Figures | fig_model_comparison.png | Core inline | Modeling notebook, milestone summaries, README-level overviews | Which validation winners look strongest once baselines and coverage risk are visible in one chart? |
| Stage-5 Performance Figures | fig_holdout_benchmark_ci.png | Core inline | Validation snapshots, deployment notes, executive summaries | Does any learned 1-minute challenger actually beat persistence on honest holdout with uncertainty shown? |
| Stage-6 Multiresolution Figures | fig_runtime_vs_gain.png | Context inline | Methodology markdowns, tradeoff sections, horizon-selection notes | Which candidates buy enough MAE improvement to justify their runtime cost? |
| Stage-7 Rollout Figures | fig_rollout_paths.png | Context inline | Rollout methodology docs, narrative result sections | Which rollout candidate tracks ramps, plateaus, and phase timing with the least visible path drift? |
| Stage-8 Horizon-Curve Figures | fig_horizon_ratio_curve.png | Core inline | Validation snapshots, report abstracts, cross-stage summaries | At which horizons does the learned stack beat persistence on the objective that actually matters there? |
| Stage-10 Forecast-Control Figures | fig_control_layer_gain_ci.png | Core inline | Operational summaries, control-stack sections, result highlights | Do hourly, phase, and nowcast updates deliver statistically meaningful control-stack gains? |
| Stage-10 Forecast-Control Figures | fig_day_ahead_refresh_policy.png | Policy inline | Optimizer-facing docs, control-policy notes, appendix markdowns | Which refresh behavior gives the best policy tradeoff between improvement and update frequency? |

## Embedded Recommended Visuals

### Stage-4 Modeling Figures: Stage-4 benchmark overview

- Why embed it: Use in modeling summaries and milestone markdowns when you need one benchmark-oriented view of the 1-minute surface.
- How to read it: Compare bar heights and annotated labels across models, feature sets, and baseline rows.
- What to look for: Whether learned models beat persistence on the selected metric, and whether any win depends on low coverage or unstable complexity.

![Stage-4 benchmark overview](../../outputs/004_modeling/commercial_facility/fig_model_comparison.png)

### Stage-5 Performance Figures: Stage-5 honest 1-minute gate

- Why embed it: Use whenever the question is whether a learned 1-minute model really beats persistence on holdout.
- How to read it: Each point is holdout MAE and the horizontal bar is the 95% moving-block bootstrap interval. Compare overlap, but rely on `holdout_inference.csv` for the exact paired delta tests.
- What to look for: Whether the learned challenger clearly separates from persistence or whether the intervals mostly overlap, which is common on strongly autocorrelated 1-minute load.

![Stage-5 honest 1-minute gate](../../outputs/005_performance/commercial_facility/latest/fig_holdout_benchmark_ci.png)

### Stage-6 Multiresolution Figures: Stage-6 compute-vs-value tradeoff

- Why embed it: Use when explaining why some learned candidates are interesting but still not worth their runtime cost.
- How to read it: The x-axis is runtime in minutes and the y-axis is MAE gain over persistence. Better candidates sit higher and further left.
- What to look for: Candidates above zero gain that do not drift into disproportionately high runtime.

![Stage-6 compute-vs-value tradeoff](../../outputs/006_multires/commercial_facility/latest/fig_runtime_vs_gain.png)

### Stage-7 Rollout Figures: Stage-7 rollout behavior

- Why embed it: Use when you need one qualitative picture of how recursive rollout candidates track the actual path over time.
- How to read it: Compare each forecast path against the actual load line over the same timestamps.
- What to look for: Divergence after the origin, missed peaks, and whether the learned candidate corrects or amplifies baseline drift.

![Stage-7 rollout behavior](../../outputs/007_rollout/commercial_facility/latest/fig_rollout_paths.png)

### Stage-8 Horizon-Curve Figures: Stage-8 horizon capability curve

- Why embed it: Use as the default cross-horizon figure because it shows where learned models cross above or below persistence on the right objective.
- How to read it: A value below 1.0 means the learned candidate beat persistence on that metric at that horizon.
- What to look for: Crossings above or below parity, especially where next-lock quality improves before profile-shape quality does.

![Stage-8 horizon capability curve](../../outputs/009_horizon_curve/commercial_facility/latest/fig_horizon_ratio_curve.png)

### Stage-10 Forecast-Control Figures: Stage-10 stacked control gains

- Why embed it: Use when you need the strongest statistical evidence that hourly, phase, and nowcast updates really improve the stack.
- How to read it: Each point is the mean gain versus the previous layer, with a bootstrap confidence interval. Positive values mean the later layer reduced error.
- What to look for: Intervals entirely above zero indicate stronger evidence that the layer helps beyond noise in a small sample of control cycles.

![Stage-10 stacked control gains](../../outputs/010_forecast_control/commercial_facility/latest/fig_control_layer_gain_ci.png)

### Stage-10 Forecast-Control Figures: Stage-10 refresh policy choice

- Why embed it: Use when documenting why the repo prefers frozen, unconditional, or triggered day-ahead refresh behavior.
- How to read it: Compare the profile-shape and lock MAE bars for the frozen, unconditional, and triggered scenarios. Lower is better for both metrics.
- What to look for: Triggered refresh should improve profile-shape error without increasing lock MAE versus the frozen path. If unconditional refresh wins but triggered does not, the trigger logic still needs work.

![Stage-10 refresh policy choice](../../outputs/010_forecast_control/commercial_facility/latest/fig_day_ahead_refresh_policy.png)

## Integrated Visuals

### 1-minute Holdout Leaderboard

- Purpose: Answer the most honest deployment question first: whether any learned 1-minute challenger actually beats persistence on holdout.
- How to read it: Lower bars are better. The persistence reference line marks the baseline the learned model must beat before it should be promoted.
- What to look for: A deployable learned winner should sit left of persistence on raw MAE, not only look good on validation or blended diagnostics.
- Source artifacts: Stage-5 holdout_evaluation.csv and deployment_recommendation.json

### Matched-Horizon Runtime vs Persistence

- Purpose: Show which Stage-6 learned candidates earn their compute cost once horizon, runtime, and persistence-relative error are considered together.
- How to read it: Lower is better on the y-axis because ratios below 1.0 beat persistence. Moving left is better on runtime, and larger markers indicate longer horizons.
- What to look for: Candidates near the lower-left frontier and, ideally, below parity. Fast candidates that still lose to persistence are not operational wins.
- Source artifacts: Stage-6 matched_horizon_metrics.csv

### Objective-Aware Horizon Ratio Curve

- Purpose: Summarize where the learned stack beats persistence or the best baseline once the metric is matched to the horizon's real operating objective.
- How to read it: Values below 1.0 are wins. The blue line compares learned winners to persistence; the orange dashed line compares them to the strongest baseline.
- What to look for: Parity crossings, especially where short horizons win on next-lock quality while day-ahead horizons win on profile shape instead.
- Source artifacts: Stage-8 horizon_curve_summary.csv

### Horizon Win Matrix

- Purpose: Separate 'where learned helps' from 'where learned looks good only on the wrong metric' by showing win/loss status across metric families.
- How to read it: Green cells mean the learned winner beats persistence for that horizon-metric pair. Red cells mean persistence still wins.
- What to look for: Concentrated green on the metric family that actually matters for that horizon, not blanket wins across every metric.
- Source artifacts: Stage-8 horizon_curve_summary.csv

### Control-Layer Error Trajectory

- Purpose: Show whether each successive forecast update layer improves the exact control cycle and the broader rolling evaluation surface.
- How to read it: The lines should move down as the stack progresses from frozen day-ahead to hourly, phase, and nowcast corrections.
- What to look for: Consistent downward error movement in both exact and rolling scopes, not just a one-window improvement.
- Source artifacts: Stage-10 control_backtest_summary.csv and rolling_control_backtest_summary.csv

### Cycle-by-Cycle Error Distribution

- Purpose: Expose variability, not only averages, so control-layer gains can be judged on stability and tail behavior.
- How to read it: Each box shows the distribution of cycle-level errors for one layer. Lower medians and tighter boxes indicate more reliable operating behavior.
- What to look for: Later layers should shift the full distribution down, not only improve the mean while leaving heavy tails intact.
- Source artifacts: Stage-10 control_backtest_by_cycle.csv and rolling_control_backtest_by_cycle.csv

### Day-Ahead Refresh Policy Comparison

- Purpose: Explain why the chosen day-ahead refresh policy is frozen, always-refresh, or triggered-refresh rather than treating refresh as a hidden rule.
- How to read it: Lower bars are better. Compare frozen, unconditional refresh, and triggered refresh on lock MAE and profile-shape MAE together.
- What to look for: Triggered refresh should capture most of the unconditional improvement without forcing unnecessary updates every cycle.
- Source artifacts: Stage-10 day_ahead_refresh_summary.csv

## Stage-7 Objective Winners

| Selection target | Winner | Metric value | Metric % | Support | Decision reason |
| --- | --- | --- | --- | --- | --- |
| endpoint_mae | hgb-balanced::avg_workday_residual | 882.409 | 40.220% | 8 origins | Lowest endpoint MAE across rollout candidates. |
| path_mae | hgb-balanced::hybrid_workday_residual | 782.772 | 39.527% | 8 origins | Lowest path MAE across rollout candidates. |
| phase_mean_mae | hgb-balanced::hybrid_workday_residual | 236.079 | 11.921% | 8 origins | Lowest 15-minute phase-average MAE across rollout candidates. |
| next_lock_mae | hgb-balanced::raw | 409.047 | 19.456% | 8 origins | Lowest next 15-minute MAE across rollout candidates. |
| profile_shape_mae | hgb-balanced::raw | 717.778 | 36.245% | 8 origins | Lowest profile-shape MAE across rollout candidates. |

## Manifest Health

| Surface | Status | Run id |
| --- | --- | --- |
| Stage-4 notebook benchmark | success | n/a |
| Stage-5 holdout gate | success | 20260321T180839930380Z |
| Stage-6 multires | success | 20260320T085821258626Z |
| Stage-7 rollout | success | 20260320T090013882130Z |
| Stage-7 challenger sweep | success | 20260320T090013545419Z |
| Stage-8 horizon curve | success | 20260320T090921574274Z |
| Stage-10 forecast control | success | 20260322T045429102555Z |
| Notebook archive | success | 20260322T042828995952Z |

## Existing Artifact Gallery

### Stage-4 Modeling Figures

These notebook-produced figures are the primary visual evidence for the Stage-4 benchmark surface. They explain how the current `1min` modeling stack is measured and where reviewers should expect forecast quality to succeed or fail.

- Figure guide: [../../outputs/004_modeling/commercial_facility/figure_guide.md](../../outputs/004_modeling/commercial_facility/figure_guide.md)
- `fig_actual_vs_predicted.png`: Actual vs predicted overlay
  - Intent: Show whether the selected validation-day forecast follows the observed load shape and turning points.
  - How to read it: Compare the learned curve and baselines against the actual load line across the same timestamps.
  - What to look for: Large misses at ramps, sustained bias above or below the actual curve, and whether the chosen model improves on persistence where operations change quickly.
  - Image: [../../outputs/004_modeling/commercial_facility/fig_actual_vs_predicted.png](../../outputs/004_modeling/commercial_facility/fig_actual_vs_predicted.png)
- `fig_error_by_hour.png`: MAE by hour of day
  - Intent: Show when during the day the forecast family is weakest or strongest.
  - How to read it: Read each line as the average absolute error for that model family at each clock hour.
  - What to look for: Error spikes during morning start-up, lunch transitions, or evening shut-down; these reveal where feature design or baseline corrections still need work.
  - Image: [../../outputs/004_modeling/commercial_facility/fig_error_by_hour.png](../../outputs/004_modeling/commercial_facility/fig_error_by_hour.png)
- `fig_model_comparison.png`: Model comparison summary
  - Intent: Give a compact benchmark comparison across the Stage-4 experiment grid and baselines.
  - How to read it: Compare bar heights and annotated labels across models, feature sets, and baseline rows.
  - What to look for: Whether learned models beat persistence on the selected metric, and whether any win depends on low coverage or unstable complexity.
  - Image: [../../outputs/004_modeling/commercial_facility/fig_model_comparison.png](../../outputs/004_modeling/commercial_facility/fig_model_comparison.png)
- `fig_day_ahead.png`: Day-ahead profile example
  - Intent: Show the 24-hour shape quality of the day-ahead extension against actual load and prior-day structure.
  - How to read it: Track whether the predicted profile captures the right daily envelope, peak timing, and trough timing even when pointwise error remains high.
  - What to look for: Profile-shape alignment, missed peak timing, and whether the forecast is useful as an operational planning surface before intraday corrections arrive.
  - Image: [../../outputs/004_modeling/commercial_facility/fig_day_ahead.png](../../outputs/004_modeling/commercial_facility/fig_day_ahead.png)

### Stage-5 Performance Figures

These figures explain how Stage-5 measures candidate quality, why coverage matters, where the promoted challenger sits relative to short-horizon baselines, and which features actually drive the learned improvement that remains after the persistence anchor.

- Figure guide: [../../outputs/005_performance/commercial_facility/latest/figure_guide.md](../../outputs/005_performance/commercial_facility/latest/figure_guide.md)
- `fig_selection_frontier.png`: Selection frontier
  - Intent: Show which Stage-5 candidates balance low error against strong validation coverage.
  - How to read it: Read left-to-right as coverage and bottom-to-top as MAE ratio versus persistence. Better candidates sit low and to the right.
  - What to look for: Candidates below 1.0 MAE ratio that also stay near full coverage; low-coverage wins should not be promoted.
  - Image: [../../outputs/005_performance/commercial_facility/latest/fig_selection_frontier.png](../../outputs/005_performance/commercial_facility/latest/fig_selection_frontier.png)
- `fig_holdout_benchmark_ci.png`: Holdout benchmark intervals
  - Intent: Show the promoted learned challenger and the short-horizon baselines with autocorrelation-aware holdout uncertainty, not just point estimates.
  - How to read it: Each point is holdout MAE and the horizontal bar is the 95% moving-block bootstrap interval. Compare overlap, but rely on `holdout_inference.csv` for the exact paired delta tests.
  - What to look for: Whether the learned challenger clearly separates from persistence or whether the intervals mostly overlap, which is common on strongly autocorrelated 1-minute load.
  - Image: [../../outputs/005_performance/commercial_facility/latest/fig_holdout_benchmark_ci.png](../../outputs/005_performance/commercial_facility/latest/fig_holdout_benchmark_ci.png)
- `fig_feature_importance.png`: Learned challenger feature importance
  - Intent: Show which predictors matter most for the current best learned short-horizon challenger.
  - How to read it: Longer bars mean permuting that feature hurts holdout MAE more. Importance is measured on the honest holdout window, so small values indicate the model depends mostly on autocorrelation and only secondarily on added features.
  - What to look for: Whether a small set of lag, phase, or profile features dominates the top of the chart and how quickly the cumulative importance concentrates.
  - Image: [../../outputs/005_performance/commercial_facility/latest/fig_feature_importance.png](../../outputs/005_performance/commercial_facility/latest/fig_feature_importance.png)

### Stage-6 Multiresolution Figures

These figures explain how Stage-6 balances runtime, stability, and persistence-relative gain when choosing a resolution-horizon winner.

- Figure guide: [../../outputs/006_multires/commercial_facility/latest/figure_guide.md](../../outputs/006_multires/commercial_facility/latest/figure_guide.md)
- `fig_runtime_vs_gain.png`: Runtime vs gain
  - Intent: Show whether a slower candidate earns enough persistence-relative improvement to justify its runtime.
  - How to read it: The x-axis is runtime in minutes and the y-axis is MAE gain over persistence. Better candidates sit higher and further left.
  - What to look for: Candidates above zero gain that do not drift into disproportionately high runtime.
  - Image: [../../outputs/006_multires/commercial_facility/latest/fig_runtime_vs_gain.png](../../outputs/006_multires/commercial_facility/latest/fig_runtime_vs_gain.png)
- `fig_resolution_pareto.png`: Resolution Pareto frontier
  - Intent: Show which eligible learned candidates are on the best observed stability-error frontier.
  - How to read it: The x-axis is fold variability and the y-axis is MAE ratio to persistence. Better candidates sit toward the lower-left frontier.
  - What to look for: Resolution choices that are both low-error and stable across folds, not just one-off low-error outliers.
  - Image: [../../outputs/006_multires/commercial_facility/latest/fig_resolution_pareto.png](../../outputs/006_multires/commercial_facility/latest/fig_resolution_pareto.png)

### Stage-7 Rollout Figures

These figures explain not only average rollout error, but also how candidate quality changes by origin and over the recursive forecast path.

- Figure guide: [../../outputs/007_rollout/commercial_facility/latest/figure_guide.md](../../outputs/007_rollout/commercial_facility/latest/figure_guide.md)
- `fig_rollout_paths.png`: Representative rollout paths
  - Intent: Show how the first selected origin evolves over the full recursive horizon for the learned candidate and baselines.
  - How to read it: Compare each forecast path against the actual load line over the same timestamps.
  - What to look for: Divergence after the origin, missed peaks, and whether the learned candidate corrects or amplifies baseline drift.
  - Image: [../../outputs/007_rollout/commercial_facility/latest/fig_rollout_paths.png](../../outputs/007_rollout/commercial_facility/latest/fig_rollout_paths.png)
- `fig_rollout_error_by_origin.png`: Endpoint error by origin
  - Intent: Show how sensitive each rollout candidate is to the starting timestamp.
  - How to read it: Each grouped bar compares endpoint absolute error across origin timestamps for the evaluated candidates.
  - What to look for: Candidates whose error remains stable across origins rather than winning only on a few convenient start times.
  - Image: [../../outputs/007_rollout/commercial_facility/latest/fig_rollout_error_by_origin.png](../../outputs/007_rollout/commercial_facility/latest/fig_rollout_error_by_origin.png)

### Stage-8 Horizon-Curve Figures

These figures explain where the current stack wins or loses as forecast horizon grows. The horizon curve is a capability envelope, not a single-model decay chart.

- Figure guide: [../../outputs/009_horizon_curve/commercial_facility/latest/figure_guide.md](../../outputs/009_horizon_curve/commercial_facility/latest/figure_guide.md)
- `fig_horizon_ratio_curve.png`: Ratio curve
  - Intent: Show learned performance relative to persistence across each horizon and objective.
  - How to read it: A value below 1.0 means the learned candidate beat persistence on that metric at that horizon.
  - What to look for: Crossings above or below parity, especially where next-lock quality improves before profile-shape quality does.
  - Image: [../../outputs/009_horizon_curve/commercial_facility/latest/fig_horizon_ratio_curve.png](../../outputs/009_horizon_curve/commercial_facility/latest/fig_horizon_ratio_curve.png)
- `fig_horizon_absolute_mae.png`: Absolute MAE surface
  - Intent: Show the absolute error tradeoffs among the learned winner, the best baseline, and persistence.
  - How to read it: Each subplot focuses on one metric family across horizons; compare the learned line to the baseline and persistence lines.
  - What to look for: Horizons where the learned candidate beats persistence but still trails a stronger baseline, and horizons where it becomes the clear winner.
  - Image: [../../outputs/009_horizon_curve/commercial_facility/latest/fig_horizon_absolute_mae.png](../../outputs/009_horizon_curve/commercial_facility/latest/fig_horizon_absolute_mae.png)

### Stage-10 Forecast-Control Figures

These figures explain whether the stacked day-ahead plus intraday update policy actually reduces operational error on shared control cycles.

- Figure guide: [../../outputs/010_forecast_control/commercial_facility/latest/figure_guide.md](../../outputs/010_forecast_control/commercial_facility/latest/figure_guide.md)
- `fig_control_lock_mae.png`: Locked-interval MAE progression
  - Intent: Show how much each control layer reduces the next locked 15-minute interval error.
  - How to read it: Each bar is the lock MAE after applying one more layer of updates to the frozen day-ahead forecast.
  - What to look for: A clear downward progression from day-ahead to hourly to phase to nowcast updates; if that pattern breaks, the control stack is not adding value.
  - Image: [../../outputs/010_forecast_control/commercial_facility/latest/fig_control_lock_mae.png](../../outputs/010_forecast_control/commercial_facility/latest/fig_control_lock_mae.png)
- `fig_control_example_cycle.png`: Example control cycle
  - Intent: Show how the full 24-hour profile changes as hourly, phase, and minute-level updates are applied.
  - How to read it: Compare the actual line with the frozen day-ahead path, then with the hourly-updated, phase-updated, and nowcast-updated paths over the same cycle.
  - What to look for: Whether intraday updates pull the forecast toward actual peaks and troughs before the next costly interval locks in, especially in the last-minute correction layer.
  - Image: [../../outputs/010_forecast_control/commercial_facility/latest/fig_control_example_cycle.png](../../outputs/010_forecast_control/commercial_facility/latest/fig_control_example_cycle.png)
- `fig_day_ahead_refresh_policy.png`: Day-ahead refresh policy comparison
  - Intent: Compare the frozen 24-hour profile with unconditional and triggered residual-refresh policies on the same exact control cycles.
  - How to read it: Compare the profile-shape and lock MAE bars for the frozen, unconditional, and triggered scenarios. Lower is better for both metrics.
  - What to look for: Triggered refresh should improve profile-shape error without increasing lock MAE versus the frozen path. If unconditional refresh wins but triggered does not, the trigger logic still needs work.
  - Image: [../../outputs/010_forecast_control/commercial_facility/latest/fig_day_ahead_refresh_policy.png](../../outputs/010_forecast_control/commercial_facility/latest/fig_day_ahead_refresh_policy.png)
- `fig_control_lock_distribution.png`: Rolling control lock-MAE distribution
  - Intent: Show whether the selected stack improves locked-interval error across a broader set of control cycles, not just the exact evaluation slice.
  - How to read it: Each box summarizes the rolling benchmark distribution for one layer's 15-minute lock MAE. Lower boxes and medians are better.
  - What to look for: A lower hourly box than day-ahead, and a lower nowcast box than phase. If distributions overlap heavily or move upward, the stacked gain is not robust.
  - Image: [../../outputs/010_forecast_control/commercial_facility/latest/fig_control_lock_distribution.png](../../outputs/010_forecast_control/commercial_facility/latest/fig_control_lock_distribution.png)
- `fig_control_layer_gain_ci.png`: Rolling layer-gain confidence intervals
  - Intent: Show the uncertainty around stacked layer gains on the rolling benchmark.
  - How to read it: Each point is the mean gain versus the previous layer, with a bootstrap confidence interval. Positive values mean the later layer reduced error.
  - What to look for: Intervals entirely above zero indicate stronger evidence that the layer helps beyond noise in a small sample of control cycles.
  - Image: [../../outputs/010_forecast_control/commercial_facility/latest/fig_control_layer_gain_ci.png](../../outputs/010_forecast_control/commercial_facility/latest/fig_control_layer_gain_ci.png)
- `fig_phase_stack_candidates.png`: Phase stack candidate frontier
  - Intent: Show which phase candidates improve lock error after the hourly stack without giving back too much profile quality.
  - How to read it: Candidates in the upper-left improve lock MAE while keeping profile regression low. The selected candidate is annotated directly on the plot.
  - What to look for: A learned candidate that clears the guard thresholds and beats the persistence passthrough baseline on the stacked surface.
  - Image: [../../outputs/010_forecast_control/commercial_facility/latest/fig_phase_stack_candidates.png](../../outputs/010_forecast_control/commercial_facility/latest/fig_phase_stack_candidates.png)

## Supporting References

- [current_validation_snapshot.md](current_validation_snapshot.md)
- [current_operating_approach.md](current_operating_approach.md)
- [model_and_blend_guide.md](model_and_blend_guide.md)
- [hypothesis.md](hypothesis.md)
