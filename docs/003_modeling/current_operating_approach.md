# Current Operating Approach

If you need the shortest high-level answer for what the repo is doing now,
start here.

This document summarizes the current evidence-backed operating policy from the
latest fully persisted Stage-10 control bundle and points to the supporting
artifacts for deeper review.

If you want the one-page generated current-state summary first, read
[current_validation_snapshot.md](current_validation_snapshot.md) before this
longer operating narrative.

## Current State

The latest fully persisted control bundle is:

- [`outputs/010_forecast_control/commercial_facility/latest`](../../outputs/010_forecast_control/commercial_facility/latest)
  (`20260322T045429102555Z`)

That bundle is current. The quality surface is strong and the promotion logic is
stricter, and it now includes both the hard dynamic minute-overlay shadow
analysis and the first soft-overlay shadow search in the same persisted bundle.
Cold Stage-10 runtime is still materially better than the old
multi-thousand-second baseline, and the latest replay is now faster again:
wall clock was `722.54s`, the dominant hotspot is `select_phase_stack_policy`
at `174.92s`, and the current applied `15m` slot is back to a distinct live
phase correction rather than hourly passthrough.

If the candidate labels below feel dense, read
[model_and_blend_guide.md](model_and_blend_guide.md) alongside this document.
It decodes label anatomy, blend wrappers, and why Stage-5, Stage-8, and
Stage-10 can honestly keep different winners.

If you need the current optimizer-facing hypothesis and delivery framing, read
[operational_hypotheses.md](operational_hypotheses.md) and
[optimizer_delivery_contract.md](optimizer_delivery_contract.md) after this
document. For the active direction, retired bets, and source credit, also read
[002_operating_direction_spec.md](../000_governance/002_operating_direction_spec.md).

## What We Learned

The repository does not support a single-model story.

The evidence supports a layered forecast-control stack:

1. Freeze a `24h` anchor before the day starts.
2. Re-evaluate that anchor on exact control cycles, not only on upstream sweep
   results.
3. Apply intraday layers only because they improve the stacked control outcome,
   not because they won in isolation.
4. Keep the `1m` layer separate from the broader rollout layers, because the
   minute nowcast still behaves differently from the `15m` and `60m` layers.

Research-aligned source credit and the formal retirement of the old
single-model direction live in
[002_operating_direction_spec.md](../000_governance/002_operating_direction_spec.md).

## Current Operating Policy

Source artifacts:

- [`control_policy.json`](../../outputs/010_forecast_control/commercial_facility/latest/control_policy.json)
- [`control_backtest_summary.csv`](../../outputs/010_forecast_control/commercial_facility/latest/control_backtest_summary.csv)
- [`control_layer_candidate_benchmarks.csv`](../../outputs/010_forecast_control/commercial_facility/latest/control_layer_candidate_benchmarks.csv)
- [`../../outputs/005_performance/commercial_facility/latest/operating_policy.json`](../../outputs/005_performance/commercial_facility/latest/operating_policy.json)
- [`../../outputs/005_performance/commercial_facility/latest/supplemental_surface_advisory.json`](../../outputs/005_performance/commercial_facility/latest/supplemental_surface_advisory.json)
- [`optimizer_delivery_contract.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_delivery_contract.json)
- [`optimizer_operational_policy.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_operational_policy.json)
- [`optimizer_dynamic_overlay_shadow_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_shadow_summary.json)
- [`optimizer_dynamic_overlay_soft_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_soft_summary.json)
- [`optimizer_dynamic_overlay_soft_candidates.csv`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_soft_candidates.csv)
- [`optimizer_delivery_uncertainty_summary.csv`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_delivery_uncertainty_summary.csv)

Current exact-control layer choices:

- Day-ahead frozen anchor: `10min/minimal/hgb-balanced::raw`
- Day-ahead refresh overlay candidate: `10min/minimal/hgb-balanced::hybrid_workday_residual`
  with current operating recommendation `triggered_refresh`
- Hourly control layer: `10min/minimal/hybrid_workday`
- Isolated phase benchmark winner: `5min/full/hybrid_workday`
- Exact stack benchmark still evaluates candidates such as
  `phase_bucket_portfolio::stack_origin_metric_policy`
- Final applied phase policy: `phase_bucket_portfolio::stack_origin_metric_policy`
  because the stack-aware phase candidate cleared the exact and rolling
  next-lock / peak / profile / optimizer guardrail bundle on the latest cold
  bundle
- Minute nowcast anchor: `curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02`
- Stage-5 minute operating role: standalone `baseline_anchor`, Stage-10 `corrective_overlay_specialist`

Exact-control results on the latest eight evaluation cycles:

| Layer | Lock MAE | Lock MAE% | Profile-Shape MAE | Profile-Shape MAE% |
|-------|----------|-----------|-------------------|--------------------|
| Frozen day-ahead | `767.411283` | `40.518170%` | `788.533702` | `41.185297%` |
| After hourly updates | `490.428482` | `25.160719%` | `626.681554` | `32.175992%` |
| After phase updates | `417.229872` | `21.443149%` | `570.445008` | `29.275415%` |
| After nowcast updates | `47.503499` | `2.459006%` | `174.956343` | `9.038661%` |

Interpretation:

- the full stacked update policy is materially better than leaving the `24h`
  forecast frozen
- the day-ahead anchor is now learned again on the exact-control surface
- the hourly layer is doing real work
- the repo still keeps a structural `15m` phase slot, and the latest exact
  stack guard re-promotes a distinct stack-aware phase correction instead of
  collapsing back to hourly passthrough
- the current exact-control minute layer is now a learned XGBoost control
  blend, even though Stage-5 holdout still does not support a
  blanket learned superiority claim at `1m`
- the broader Stage-5 supplemental advisory surface now says the learned
  `1m` challenger does beat persistence on stitched validate-walkforward plus
  holdout rows, but that support is concentrated in `transition_only` and
  `transition_active`; the strongest supportive diagnostic slice is
  `actual_ramp_band=high_ramp` at a `0.878199` MAE ratio to persistence, while
  the narrow canonical `none_inactive` holdout slice remains baseline-led
- the Stage-10 code path now uses that broader advisory surface only as a
  near-tie breaker for minute-overlay selection; when exact-control candidates
  are materially separated, the held-out control metric still wins outright
- the same bundle now also carries a shadow-only dynamic minute controller; on
  the latest replay it marked only `32 / 776` rows as strategically eligible,
  and enforcing that gate would have worsened all-interval selected absolute
  error from `47.503499` to `417.163949` while leaving next-lock MAE and
  peak-hit rate unchanged, so the repo correctly keeps that controller in
  diagnostic mode instead of letting it alter live layer resolution
- the same bundle now also carries the first soft minute-overlay shadow search;
  it evaluated `29` weight policies and still selected pure nowcast
  (`soft_overlay_sw100_bw100`) as the best admissible policy, so the repo now
  has positive evidence that the current learned minute layer does not benefit
  from background softening on this replay surface either
- the Stage-10 bundle is now also the repo's pre-optimizer delivery surface:
  it emits selected interval forecasts, calibrated residual bands, and contract
  metadata instead of only aggregate replay summaries

Optimizer-facing exact-control slices from the same latest summary:

- next-lock MAE:
  `252.606386` -> `132.868618` -> `132.868618` -> `32.483936`
- peak-value MAE:
  `2908.519733` -> `571.351469` -> `324.703109` -> `31.995877`
- peak-interval hit rate:
  `0.00` -> `0.00` -> `0.125` -> `0.875`

Interpretation:

- the hourly layer materially improves locked-interval, next-lock, and
  peak-value error versus the frozen day-ahead path
- the phase slot now adds real stack value on lock, profile-shape, and
  peak-value behavior, even though it remains flat on next-lock MAE in the
  current exact-control surface
- the minute nowcast is still the layer that materially recovers both next-lock
  error and peak capture
- the scale-aware next-lock uncertainty override materially narrowed the latest
  exact-control `95%` next-lock band to `40.425144%`, but the interval is still
  conservative and still only backed by eight next-lock evaluation rows

## Rolling Benchmark

The exact-control replay is the operational surface. Stage-10 now supplements
it with a full out-of-sample rolling benchmark across all eligible validate/test
origins on the configured schedule. That broader benchmark is the right answer
to "simulate the live feed and compare it across the available held-out history"
without contaminating the report with train-split optimism.

- calibration cycles: `16`
- evaluation cycles: `16`
- scope artifacts:
  - [`rolling_control_scope_summary.csv`](../../outputs/010_forecast_control/commercial_facility/latest/rolling_control_scope_summary.csv)
  - [`rolling_control_layer_inference.csv`](../../outputs/010_forecast_control/commercial_facility/latest/rolling_control_layer_inference.csv)

Rolling evaluation means:

| Layer | Lock MAE | Profile-Shape MAE |
|-------|----------|-------------------|
| Frozen day-ahead | `763.962699` | `786.255244` |
| After hourly updates | `492.201440` | `626.787911` |
| After phase updates | `270.220472` | `432.403931` |
| After nowcast updates | `47.500033` | `175.213594` |

Key inference:

- hourly vs day-ahead lock gain:
  `271.761259`, 95% CI [`146.509717`, `379.601589`], `p=0.0000`
- hourly vs day-ahead profile gain:
  `159.467333`, 95% CI [`80.165916`, `228.327102`], `p=0.0000`
- phase vs hourly lock gain:
  `221.980968`, 95% CI [`187.041677`, `254.189649`], `p=0.0000`
- phase vs hourly profile gain:
  `194.383980`, 95% CI [`157.199807`, `229.052739`], `p=0.0000`
- phase vs hourly next-lock gain:
  `0.000000`, 95% CI [`0.000000`, `0.000000`], `p=1.0000`
- nowcast vs phase lock gain:
  `444.737384`, 95% CI [`408.187377`, `482.212764`], `p=0.0000`
- nowcast vs phase next-lock gain:
  `82.168240`, 95% CI [`47.275961`, `118.149340`], `p=0.0000`

Interpretation:

- the hourly layer is robust on the broader rolling surface, not just on the
  exact 8-cycle replay
- the broader rolling benchmark now also supports a distinct applied phase
  correction on lock, profile-shape, and optimizer score, even though next-lock
  remains flat versus hourly
- the minute nowcast remains the strongest stack improvement layer, and the
  current rolling replay also supports the learned exact-control minute policy
- the minute operating policy is now explicit: Stage-5 stays baseline-led for
  standalone `1m`, while Stage-10 is allowed to use learned minute overlays as
  corrective specialists when the control replay supports them

## Current Resolution Policy

The pipeline supports sub-minute cadences, but the current operating control
surface does not use them as its default resolution.

Source artifacts:

- [`../../outputs/006_multires/commercial_facility/latest/selection_summary.csv`](../../outputs/006_multires/commercial_facility/latest/selection_summary.csv)
- [`../../outputs/006_multires/commercial_facility/latest/matched_horizon_metrics.csv`](../../outputs/006_multires/commercial_facility/latest/matched_horizon_metrics.csv)
- [`control_policy.json`](../../outputs/010_forecast_control/commercial_facility/latest/control_policy.json)

Current evidence-backed policy:

- supported ingest/comparison resolutions remain
  `1s`, `5s`, `10s`, `30s`, `1min`, `5min`, `10min`, `15min`
- the default materialized pipeline surface remains
  `1min`, `5min`, `10min`, `15min`
- the current Stage-10 actual resolution is still `1min`
- the latest Stage-6 matched-horizon winners that cleared the repo's
  coverage/stability/runtime gates still resolve to `1min` at both `15m` and
  `60m`
- `30s` remains the only sub-minute cadence with meaningful current evidence,
  but it is still exploratory: the latest
  `30s/curated/xgb-balanced/recursive` challenger reached a `0.827315` MAE
  ratio to persistence at `15m`, yet it still failed the current operating
  gates with `eligible=False` and `fold_std_mae_ratio=0.287154` against a
  `0.200000` stability cap
- there is no current persisted evidence that justifies promoting `1s`, `5s`,
  or `10s` below the present `1min` control contract
- the repo now also carries a dedicated `subminute_focus` Stage-6 profile so
  future `30s` vs `1min` investigations can stay centralized and reproducible
  without changing the current operating default

Interpretation:

- `1min` is not being kept out of habit; it is the current operating winner once
  the repo's selection gates, replay surface, and optimizer-facing contract are
  considered together
- sub-minute support is valuable for experimentation, but it should currently be
  treated as exploratory rather than as the default optimizer-facing feed

## Delivery Contract

Stage-10 now emits a delivery-shaped surface in addition to the aggregate
backtest summaries:

- [`optimizer_delivery_contract.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_delivery_contract.json)
- [`optimizer_operational_policy.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_operational_policy.json)
- [`optimizer_delivery_preview.csv`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_delivery_preview.csv)
- [`optimizer_delivery_serving_preview.csv`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_delivery_serving_preview.csv)
- [`optimizer_dynamic_overlay_shadow_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_shadow_summary.json)
- [`optimizer_dynamic_overlay_soft_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_soft_summary.json)
- [`optimizer_dynamic_overlay_soft_candidates.csv`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_dynamic_overlay_soft_candidates.csv)
- [`optimizer_delivery_uncertainty_summary.csv`](../../outputs/010_forecast_control/commercial_facility/latest/optimizer_delivery_uncertainty_summary.csv)

Current delivery readings from the latest persisted bundle:

- cadence: `15 minutes`
- contract version: `1.2`
- layer priority: `nowcast -> phase -> hourly -> day_ahead`
- current applied phase policy:
  `phase_stack_guard_policy = phase_candidate`
- all-interval empirical coverage:
  `80% band = 0.934278`, `95% band = 0.984536`
- next-lock empirical coverage:
  `80% band = 0.875000`, `95% band = 0.875000`
- actual-peak empirical coverage:
  `80% band = 0.875000`, `95% band = 1.000000`
- dynamic minute-overlay shadow verdict:
  `keep_shadow_mode`; hard enforcement would raise all-interval selected
  absolute error from `47.503499` to `417.163949`
- soft minute-overlay shadow verdict:
  `keep_pure_nowcast_shadow`; the best admissible soft policy remained pure
  nowcast (`soft_overlay_sw100_bw100`) on the same replay surface

Interpretation:

- the repo now has an explicit pre-optimizer contract instead of only a
  notebook/report story
- the preview rows now also carry freshness, confidence, provenance, and
  executable fallback-resolution fields directly, rather than forcing a
  downstream consumer to reconstruct them from side artifacts
- the serving preview strips out replay-only truth columns while keeping the
  same contract fields that a downstream optimizer would consume, including the
  interval operating regime and the dynamic minute-controller audit fields
- the current empirical bands are conservative, which is acceptable for a first
  risk surface but should not be oversold as a final probabilistic model
- the new operational policy artifact makes the CPU-safe fallback path and the
  optional accelerated path explicit instead of implicit
- a one-cycle live-resolution simulation on the latest bundle behaved as
  intended: `0m -> nowcast`, `+10m -> phase candidate`, `+45m -> hourly`,
  `+120m -> day_ahead`
- the Stage-5 supplemental advisory surface is now part of that honesty story:
  it broadens the `1m` evidence base without rewriting the canonical holdout
  gate
- the dynamic minute controller is now also part of that honesty story: the
  repo persists the shadow-vs-enforced counterfactual directly and keeps the
  controller shadow-only when the replay says a hard gate would hurt

## Day-Ahead Policy

Two statements are now simultaneously true:

- the best standalone `1440m` learned rollout is still
  `10min/minimal/hgb-balanced::raw`
- the exact-control frozen anchor currently promoted inside Stage-10 is also
  `10min/minimal/hgb-balanced::raw`

That means the broader out-of-sample control replay now agrees with the
standalone Stage-7 `24h` profile winner on the frozen day-ahead anchor itself.
The residual model remains useful as a refresh overlay, not as the primary
frozen `24h` path.

Refresh overlay evidence:

- [`day_ahead_refresh_summary.csv`](../../outputs/010_forecast_control/commercial_facility/latest/day_ahead_refresh_summary.csv)
- [`day_ahead_refresh_decisions.csv`](../../outputs/010_forecast_control/commercial_facility/latest/day_ahead_refresh_decisions.csv)

Latest refresh results:

- frozen day-ahead:
  - `lock_mae = 767.411283`
  - `profile_shape_mae = 788.533702`
- unconditional refresh:
  - `lock_mae = 606.603723`
  - `profile_shape_mae = 701.862380`
- triggered refresh:
  - `lock_mae = 655.385169`
  - `profile_shape_mae = 732.516445`
  - mean updates per cycle: `8.750`
  - evaluation trigger rate: `0.3804347826`
  - selected trigger mode: `residual_or_activity_active_or_transition`
  - retained unconditional profile gain: `64.63%`
  - retained unconditional lock gain: `69.66%`
- rolling refresh comparison:
  - unconditional refresh:
    `lock_mae = 608.635032`, `profile_shape_mae = 704.113762`
  - triggered refresh:
    `lock_mae = 649.485955`, `profile_shape_mae = 730.181648`
  - rolling trigger rate: `0.3838028169`
  - selected rolling trigger mode: `residual_or_activity_active_or_transition`

Interpretation:

- the learned residual refresh path is useful
- the trigger is no longer always-on, but it still leaves too much of the
  unconditional benefit on the table
- current operating recommendation is now `triggered_refresh` because the
  broader rolling benchmark and exact-control scope both keep enough of the
  unconditional gain while staying inside the configured trigger-rate band

## Short-Horizon Policy

Short horizons remain objective-specific:

- `1440m`: optimize for profile shape
- `60m` and `15m`: optimize for correction value before the next locked interval
- `1m`: treat as a nowcast policy, not just another rollout horizon

Current horizon summary:

- [`horizon_curve_summary.csv`](../../outputs/009_horizon_curve/commercial_facility/latest/horizon_curve_summary.csv)
- [`horizon_curve_summary.md`](../../outputs/009_horizon_curve/commercial_facility/latest/horizon_curve_summary.md)

Current takeaways:

- `1440m` beats persistence on profile-shape objective
- `60m` beats persistence on correction-window objective
- `15m` exact-control candidate benchmarking still identifies a learned
  stack-aware phase portfolio:
  `phase_bucket_portfolio::stack_origin_metric_policy`
- the latest persisted Stage-10 bundle now applies that distinct
  `15m` correction, and the rolling-support guard also keeps it live on the
  current cold bundle
- `1m` learned superiority is still not supported by the current Stage-5
  holdout evidence, but the exact-control Stage-10 nowcast surface currently
  selects a learned XGBoost control blend over persistence by a large
  operational margin at the stacked control level
- the horizon objective winner at `15m` is still learned on the Stage-8 surface,
  and the latest persisted Stage-10 bundle now keeps that structural
  `15m` slot live in the applied stack

## Execution Policy

The repo is now runtime-aware as well as model-aware.

- high-capacity x64 hosts use stage-specific parallel plans for Stage-5, Stage-7,
  and Stage-10 rather than a one-size-fits-all worker cap
- optional XGBoost candidates only appear when the acceleration extra is
  installed and the runtime probe confirms a usable device
- ARM64 or non-accelerated teammate machines stay on the conservative CPU-safe
  path without needing different commands or code branches

That keeps the repo portable across the team while still letting the stronger
host do the more expensive search and replay work efficiently.

Latest cold-runtime readout from [`runtime_summary.json`](../../outputs/010_forecast_control/commercial_facility/latest/runtime_summary.json):

- wall clock: `722.537850` seconds
- replay seconds: `412.264106`
- evaluation seconds: `307.995991`
- artifacts seconds: `1.066647`
- longest step: `select_phase_stack_policy` at `174.917094` seconds
- current runtime truth: Stage-10 is still materially better than the older
  multi-thousand-second baseline, but cold replay remains expensive and
  stack-aware phase selection is now the main optimization target

## Visuals

> **Note:** The figures below are generated by running Stage-8 and Stage-10 and
> are not stored in version control. Run the relevant stage scripts to produce
> them locally.

Stage-8 capability view:

![Horizon capability curve](../../outputs/009_horizon_curve/commercial_facility/latest/fig_horizon_ratio_curve.png)

What to look for:

- where the ratio falls below `1.0`, the learned policy beats its selected
  baseline
- where it stays above `1.0`, the baseline still owns that horizon/objective

Stage-10 control stack:

![Control lock-MAE reduction](../../outputs/010_forecast_control/commercial_facility/latest/fig_control_lock_mae.png)

What to look for:

- each layer should reduce lock MAE versus the previous layer
- if a layer barely moves the curve, it may still be winning locally while not
  adding much stack-level value

Stage-10 example cycle:

![Example control cycle](../../outputs/010_forecast_control/commercial_facility/latest/fig_control_example_cycle.png)

Stage-10 refresh comparison:

![Day-ahead refresh comparison](../../outputs/010_forecast_control/commercial_facility/latest/fig_day_ahead_refresh_policy.png)

Detailed stage-specific figure explanations live in:

- [`outputs/009_horizon_curve/commercial_facility/latest/figure_guide.md`](../../outputs/009_horizon_curve/commercial_facility/latest/figure_guide.md)
- [`outputs/010_forecast_control/commercial_facility/latest/figure_guide.md`](../../outputs/010_forecast_control/commercial_facility/latest/figure_guide.md)
- [`outputs/010_forecast_control/commercial_facility/latest/current_evidence_index.md`](../../outputs/010_forecast_control/commercial_facility/latest/current_evidence_index.md)

## How To Navigate The Repo

When the question is "what are we doing now?", use this order:

1. Read this document.
2. Read [stage_map.md](../002_pipeline/stage_map.md) if the numbered stages still feel opaque.
3. Read [`control_policy.json`](../../outputs/010_forecast_control/commercial_facility/latest/control_policy.json).
4. Read [`control_backtest_summary.md`](../../outputs/010_forecast_control/commercial_facility/latest/control_backtest_summary.md).
5. Read [`horizon_curve_summary.md`](../../outputs/009_horizon_curve/commercial_facility/latest/horizon_curve_summary.md).
6. Only then drop into the older reports for provenance.

## Current Next Step

The next highest-value work is now narrower and more evidence-constrained.

It is:

- reduce `select_phase_stack_policy` further, because the latest cold bundle
  still spends most of its runtime budget there even after the phase layer
  re-earned promotion
- keep the minute controller layered but move from hard regime gating toward a
  softer dynamic-weighting or trust-weighting experiment only as shadow
  analysis; the persisted hard-gate and soft-search summaries both say the
  active pure-nowcast policy should stay unchanged for now
- tighten next-lock uncertainty without overclaiming, because the current
  control-surface calibration still only has `8` next-lock evaluation rows

## Repository Hygiene

Old superseded dated runs were pruned with
[`scripts/tooling/cleanup_outputs.py`](../../scripts/tooling/cleanup_outputs.py).
The latest applied cleanup report is:

- [`personal/output_cleanup_report.md`](../../personal/output_cleanup_report.md)
