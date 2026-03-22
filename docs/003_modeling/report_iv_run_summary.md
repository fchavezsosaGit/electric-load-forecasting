# Report IV Run Summary (Current Provenance Refreshed 2026-03-20)

> Historical scope note:
> This document preserves the Report IV and notebook-era provenance story.
> Current operating truth is governed by
> [current_validation_snapshot.md](current_validation_snapshot.md),
> [current_operating_approach.md](current_operating_approach.md), and
> [002_operating_direction_spec.md](../000_governance/002_operating_direction_spec.md).

This document records the latest executed `1min` Minimum Viable Product (MVP) run and its measured outcomes.
It is the factual bridge between implementation specs and report-ready interpretation.

Current-note rule:
- treat the provenance block above and
  [`current_validation_snapshot.md`](current_validation_snapshot.md) as the
  canonical one-page latest-state summary
- treat
  [`current_visualization_guide.md`](current_visualization_guide.md) as the
  canonical markdown visual-reading and embed guide
- treat the interactive
  [validation dashboard](../../outputs/reports/commercial_facility/latest/validation_dashboard.html)
  as the canonical integrated HTML visual surface
- treat
  [`current_operating_approach.md`](current_operating_approach.md) as the
  canonical longer latest-state interpretation
- use [model_and_blend_guide.md](model_and_blend_guide.md) whenever the current
  candidate labels or wrapper names are not immediately clear
- older bundle ids and timings that still appear later in this document are
  retained as historical narrative unless explicitly restated in the provenance block

## Recommended Visual Anchors

When this summary is quoted into other markdowns, start with these inline
visuals:

- `fig_holdout_benchmark_ci.png`: the honest `1m` promotion gate. Use it for
  the question "does any learned challenger really beat persistence on holdout?"
- `fig_horizon_ratio_curve.png`: the default cross-horizon figure. Use it when
  you need one view of where learned candidates beat or lose to persistence on
  the objective that matters at that horizon.
- `fig_control_layer_gain_ci.png`: the strongest operational-lift figure. Use
  it when the claim is about stacked hourly, phase, and nowcast updates helping
  the optimizer-facing forecast.
- `fig_model_comparison.png`: benchmark context for Stage-4. Keep it inline in
  the notebook and modeling summaries when you need to show validation ranking
  without pretending that validation alone is deployment evidence.

For the maintained per-figure reading guide, use
[`current_visualization_guide.md`](current_visualization_guide.md). For the
interactive integrated surface, use the
[validation dashboard](../../outputs/reports/commercial_facility/latest/validation_dashboard.html).

![Stage-5 honest 1-minute gate](../../outputs/005_performance/commercial_facility/latest/fig_holdout_benchmark_ci.png)

![Stage-8 horizon capability curve](../../outputs/009_horizon_curve/commercial_facility/latest/fig_horizon_ratio_curve.png)

![Stage-10 stacked control gains](../../outputs/010_forecast_control/commercial_facility/latest/fig_control_layer_gain_ci.png)

## Run Provenance

- Current integrated quick E2E:
  `python scripts/run_e2e.py --mode quick --with-multires --with-rollout --with-rollout-sweep --with-horizon-curve --with-forecast-control`
- Notebook validation: `python scripts/validate_notebooks.py`
- Full test suite: `pytest -q`
- Stage-4 artifact directory: `outputs/004_modeling/commercial_facility/`
- Latest Stage-5 artifact directory:
  `outputs/005_performance/commercial_facility/20260320T101213998808Z/`
- Latest Stage-6 smoke artifact directory:
  `outputs/006_multires/commercial_facility/20260320T085821258626Z/`
- Latest Stage-7 challenger sweep:
  `outputs/007_rollout/commercial_facility/challenger_sweeps/20260320T090013545419Z/`
- Focused `60m` Stage-6 reference directory:
  `outputs/006_multires/commercial_facility/20260310T005916684602Z/`
- Latest notebook archive:
  `outputs/008_notebook_runs/commercial_facility/20260322T042828995952Z/`
- Latest H5 horizon characterization:
  `outputs/009_horizon_curve/commercial_facility/20260320T090921574274Z/`
- Latest forecast-control backtest:
  `outputs/010_forecast_control/commercial_facility/20260322T030301040853Z/`

Latest integrated quick E2E timing snapshot:
- pipeline: `1951.19s`
- notebooks: `455.41s`
- pytest: `248.47s`
- total: `2655.07s`

Current Stage-10 operating stack from the latest persisted bundle:
- day-ahead frozen anchor: `10min/minimal/hgb-balanced::raw`
- hourly layer: `10min/minimal/hybrid_workday`
- phase slot currently applied as hourly passthrough:
  `10min/minimal/hybrid_workday`
- minute nowcast:
  `curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02`

## Artifact Integrity

- `metrics_overall.csv`: `57` rows
- `metrics_by_day_class.csv`: `98` rows
- `metrics_by_hour.csv`: `1176` rows
- `run_manifest.json`: present (`has_test=true`, `eval_split=validate`,
  `resolutions=['1min']`, `total_experiments=57`)
- PNG files: all four required figures present and validated through the notebook
  archive manifest
  - `fig_actual_vs_predicted.png`: `1990x772`
  - `fig_error_by_hour.png`: `1792x814`
  - `fig_model_comparison.png`: `1966x1134`
  - `fig_day_ahead.png`: `1990x772`
- Metric percentage basis: `100 * error / mean(abs(actual_load))` over valid rows

## Key Metrics (Validate)

- Raw MAE-best validate model: `feature_set=full_stable`, `model=ridge`,
  `params=strong`
  - MAE: `502.5372` (`13.1423%`)
  - RMSE: `949.1225` (`24.8214%`)
  - Coverage: `1612 / 4296 = 0.3752`
- Coverage-selected holdout model: `feature_set=curated`, `model=ridge`,
  `params=strong`
  - MAE: `596.7663` (`15.0504%`)
  - RMSE: `1141.1234` (`28.7790%`)
  - Coverage: `4152 / 4296 = 0.9665`
- Validate persistence baseline:
  - MAE: `594.7170` (`14.5976%`)
  - RMSE: `1195.9468` (`29.3550%`)
- Coverage-selected vs persistence improvement:
  - MAE: `-0.34%` (`596.7663` vs `594.7170`)
  - RMSE: `+4.59%` (`1141.1234` vs `1195.9468`)
- Raw-best vs persistence (pre-coverage-guard, 37.5% coverage -- not usable):
  - MAE: `+15.50%` (`502.5372` vs `594.7170`)
  - RMSE: `+20.64%` (`949.1225` vs `1195.9468`)

## Hypothesis Snapshot

- H1 (workday signal): evaluated per spec via `h1_control` (`feature_set=temporal_no_workday`) vs `minimal/ridge/medium`.
  - `h1_control` MAE: `679.0109`
  - `minimal/ridge/medium` MAE: `651.7676`
  - Observed delta: `+4.01%` (below `>=10%` target)
- H2 (lag/rolling value): evaluated per spec via `temporal/ridge/medium` vs `curated/ridge/medium` by RMSE.
  - `temporal/ridge/medium` RMSE: `1197.0998`
  - `curated/ridge/medium` RMSE: `1134.0089`
  - Observed delta: `+5.27%` (below `>=8%` target)
- H3 (resolution tradeoff): still deferred inside the canonical `1min` MVP notebook,
  but the Stage-6 multiresolution runtime is implemented and now has both learned and
  baseline winners outside the notebook. The latest smoke run keeps persistence at
  `15m`, while focused/candidate sweeps still show learned winners at `30m`, `60m`,
  and `120m`.
- H4 (nonlinear behavior, exploratory): evaluated via ridge/hgb grid comparisons; no hard pass/fail threshold
- H5 (horizon degradation): Stage-8 horizon curve is executed. The current objective-aware envelope now separates short-horizon next-lock quality from day-ahead profile quality: `15m` and `60m` beat persistence on next-lock MAE, `1440m` beats persistence on profile-shape MAE, and the remaining loss is the `1m` holdout anchor

## Holdout (Test) Readout

- One-shot holdout row exists for the coverage-selected model (`curated/ridge/strong`).
- Selected model holdout: MAE `213.2404` (`10.2868%`), RMSE `298.4573` (`14.3977%`).
- Persistence holdout: MAE `173.7241` (`8.3805%`), RMSE `270.9570` (`13.0664%`).
- **Persistence beats the selected model on test MAE and RMSE** (173.7 vs 213.2 W,
  271.0 vs 298.5 W).
- Validate-to-test MAE shift for selected model: `-64.27%` (test period easier than validate period).
- Validate-to-test MAE shift for persistence: `-70.79%` (persistence itself drops from 594.7 to 173.7 W).
- Root cause: test period (Dec 26-28) is 3 quiet post-Christmas holidays; validation has 2 working days + 1 holiday.
- Day-ahead extension row exists (`experiment=day_ahead`).
- Day-ahead row metrics: MAE `2665.8329` (`65.4041%`), RMSE `3949.8697` (`96.9070%`).

## Interpretation

- Implementation intent is met: deterministic run pipeline, artifacts, notebook
  validation, and holdout protocol are operational.
- Latest integrated quick E2E is green end to end:
  - pipeline: `7017.78s`
  - notebooks: `548.56s`
  - pytest: `281.63s`
  - total: `7847.97s`
- Hypothesis targets are outcomes, not implementation gates:
  - H1 target not achieved in this run.
  - H2 target partially improved but below threshold.
  - H5 characterization is now implemented and measured; there is no single crossover horizon.
- These findings indicate model-quality work remains horizon-specific; the core repo
  behavior is no longer blocked by infrastructure drift.

## Stage-5 Performance Extension (2026-03-11)

Stage-5 artifacts now live in timestamped run directories under
`outputs/005_performance/commercial_facility/`, with `outputs/005_performance/commercial_facility/latest/` mirroring the
newest successful run. They include preflight audit, walk-forward fold results,
residual ablation, HGB coordinate search, and causal blend guardrail diagnostics.
The Stage-5 results are summarized inside `notebooks/003_modeling.ipynb`; the
earlier standalone performance-notebook design was retired.
Current runtime behavior also writes:
- `coverage_audit.csv` for split-level feature-set coverage
- `promotion_candidate.json` for the exact Stage-5 scoreboard winner
- `holdout_evaluation.csv`, `holdout_predictions.csv`, and `deployment_recommendation.json`
  for the promoted-candidate vs baseline holdout decision
- `holdout_inference.csv` plus `fig_holdout_benchmark_ci.png` for moving-block
  bootstrap confidence intervals and paired holdout significance checks
- `feature_importance_permutation.csv`, `feature_importance_summary.json`, and
  `fig_feature_importance.png` for learned-challenger interpretation
- `holdout_segment_evaluation.csv` for regime-sliced holdout diagnostics
- `adaptive_hgb_screen.csv` for bounded centralized HGB screening when enabled
- `holdout_registry.csv` for cross-run Stage-5 learned holdout winners and saved blend settings
- `blend_finalists.csv` for the best validation-selected guarded blend config in each shortlisted Stage-5 learned family

Key outcomes from the latest targeted Stage-5 rerun
(`outputs/005_performance/commercial_facility/20260319T161606310469Z/`):
- Best overall fold result:
  `curated_ramp/xgb-balanced/residual+blend` with
  `fold_mean_mae_ratio=0.938920` and `mean_coverage=0.995486`.
- The quick `1m` surface is now broader and evidence-dense:
  `minimal_phase_anchor`, `full_stable`, `curated_ramp`, and `minimal_phase`
  with frontier HGB plus optional XGBoost candidates instead of the old
  two-model smoke path.
- The strongest learned short-horizon challenger is currently ramp-aware rather than
  high-capacity; this latest run still did not justify operational promotion.
- The final holdout still does **not** justify replacing persistence:
  - promoted learned MAE: `175.055450` (`8.444727%`)
  - persistence MAE: `173.724099` (`8.380502%`)
  - anchored_workday MAE: `257.954664` (`12.443809%`)
  - arima MAE: `863.113597` (`41.636855%`)
  - holt_damped MAE: `864.439892` (`41.700836%`)
  - learned/persistence MAE ratio: `1.007664`
- The new paired holdout inference now makes the `1m` conclusion explicit rather than
  qualitative:
  - MAE delta versus persistence: `+1.331351 W`
  - 95% moving-block bootstrap CI: `[+0.401274, +2.225375] W`
  - inferred block length: `15` minutes
  - one-sided p-value for the claim "learned MAE < persistence MAE": `1.0000`
  - exact conclusion: `1m` learned superiority is not supported by the current
    holdout evidence
- The RMSE story is more nuanced:
  - RMSE delta versus persistence: `-2.060269 W`
  - 95% moving-block bootstrap CI: `[-3.150256, -0.942326] W`
  - interpretation: the learned challenger reduces larger misses a little, but not
    enough to overcome its worse MAE and retain operational promotion
- The new feature-importance surface closes the interpretation loop:
  - `lag_1` remains the dominant feature
  - the top 5 features contribute `94.19%`
  - the learned gain is concentrated in autocorrelation plus a narrow set of ramp/profile
    features, not in broad feature-set diversity
- Stage-5 operational conclusion remains unchanged:
  `1m` learned superiority is not supported by the current holdout evidence;
  persistence still wins the 1-minute holdout, while `curated_ramp` remains the
  strongest measured learned Stage-5 challenger.
- The new holdout registry preserves the stronger historical learned winner
  `full_stable/hgb-frontier-lr010-l2001/raw+blend` at `163.003809` (`7.863352%`)
  so downstream nowcast replay is no longer limited to the latest quick run.
- The new `blend_finalists.csv` artifact now preserves phase-aware blended
  candidates such as `minimal_phase_anchor/hgb-frontier-lr010-leaf100/raw+blend`
  for exact-control benchmarking without changing the Stage-5 holdout promotion rule.
- `holdout_segment_evaluation.csv` is now generated, but the present holdout slice is
  not regime-diverse enough to make the segmented readout highly diagnostic yet.

### Threats to Validity

- The canonical modeling window is still short: 3 validation days and 3 holdout days.
- Christmas and the immediate post-Christmas period materially distort the difficulty
  of both validation and holdout windows.
- All findings remain single-facility evidence for `commercial_facility`.
- The new moving-block bootstrap improves statistical rigor, but it cannot create more
  regime diversity than exists in the underlying holdout slice.
- An expanded walk-forward profile (`n_folds=12, val_window_days=1`) is available via
  `config/modeling.toml` to produce per-day MAE estimates across days 17-28, providing
  finer-grained stability diagnostics than the default 5x2-day layout.
- ARIMA(1,1,1) is included as a classical benchmark alongside Holt-damped ETS; both
  are evaluated on the holdout to contextualize the difficulty of short-horizon load
  forecasting against traditional time-series methods.

## Stage-6 and Stage-7 Runtime Snapshot (2026-03-11)

- Latest integrated quick post-MVP sweep:
  - `python scripts/run_e2e.py --mode quick --with-multires --with-rollout --with-rollout-sweep --with-horizon-curve --with-forecast-control`
  - pipeline: `7017.78s`
  - notebooks: `548.56s`
  - pytest: `281.63s`
  - total: `7847.97s`
- Latest Stage-6 smoke run
  (`outputs/006_multires/commercial_facility/20260311T081905380560Z/`):
  - `matched_horizon_15m` winner: baseline persistence
  - `matched_horizon_60m` winner: learned
    `1min/minimal/hgb-balanced/recursive`
  - `60m` winner endpoint/path MAE:
    `2033.943270` (`38.744315%`) / `1248.451426` (`32.021254%`)
- Focused `60m` rerun remains the stronger Stage-6 hourly reference
  (`outputs/006_multires/commercial_facility/20260310T005916684602Z/`):
  - `matched_horizon_60m` winner: learned `5min/minimal/hgb-balanced/recursive`
  - endpoint/path MAE:
    `1148.166851` (`42.691558%`) / `1151.446627` (`36.611950%`)
- Latest targeted Stage-7 challenger sweep:
  - `python run_pipeline.py --stage rollout_sweep`
  - pipeline: `261.69s`
- Latest targeted Stage-7 rollout:
  - `python run_pipeline.py --stage rollout`
  - pipeline: `25.81s`
- Stage-6 targeted candidate selection is mixed rather than uniformly persistence-led:
  - latest smoke run (`outputs/006_multires/commercial_facility/20260311T081905380560Z/selection_summary.csv`)
    - `matched_horizon_15m` winner: baseline persistence
    - `matched_horizon_60m` winner: learned model at `1min/minimal/hgb-balanced/recursive`
  - candidate sweep (`outputs/006_multires/20260307T133220706885Z/selection_summary.csv`)
    - `matched_horizon_30m` winner: learned model at `5min/curated/hgb-balanced/direct_endpoint`
    - `matched_horizon_120m` winner: learned model at `5min/curated/hgb-balanced/recursive`
  - focused 60-minute run (`outputs/006_multires/commercial_facility/20260310T005916684602Z/selection_summary.csv`)
    - `matched_horizon_60m` winner: learned model at `5min/minimal/hgb-balanced/recursive`
    - endpoint/path MAE: `1148.166851` (`42.691558%`) / `1151.446627` (`36.611950%`)
  - interpretation: the current Stage-6 surface is working and now has verified learned
    winners at `30m`, `60m`, and `120m`, while the latest smoke run keeps
    persistence at `15m`. `60m` still favors the compact `minimal` set over the
    richer `full_stable` and `regime_profile` challengers in the latest focused
    rerun.
- Stage-7 rollout selection is now horizon-aware:
  - Stage-7 reuses a Stage-6 learned winner only when `winner_horizon_minutes` exactly matches the requested rollout horizon
  - Stage-7 further requires `winner_forecast_strategy=recursive`; direct endpoint
    winners are endpoint-only and are not promoted into rollout
  - selection now resolves in this order: explicit rollout candidate overrides, explicit `--selection-run-id`, Stage-7 `challenger_sweep_registry.csv`, Stage-6 `winner_registry.csv`, legacy `latest/selection_summary.csv`, Stage-7 `rollout_registry.csv`, then `config/multires.toml`
  - Stage-7 fallback is now objective-aware through `selection_target` (`path_mae` or `endpoint_mae`)
  - otherwise it falls back to the explicit rollout candidate in `config/multires.toml` and records the reason in `selection_context.json`
- Stage-7 challenger selection is now explicit through `python run_pipeline.py --stage rollout_sweep`:
  - the sweep ranks prior learned rollout candidates at the requested horizon using
    `rollout_registry.csv` evidence
  - all candidates in one sweep now share the same sampled origin timestamps across
    resolutions; the sweep writes `shared_origins.csv`, and each rollout candidate run
    writes `selected_origins.csv`
  - at hourly horizons, the sweep can now synthesize cross-candidate portfolio
    policies from the shared-origin by-origin artifacts and writes
    `portfolio_policy_candidates.json` plus `portfolio_policy_by_origin.csv`
  - it writes `candidate_results.csv`, `challenger_summary.md`, and
    `recommended_candidate.json` under
    `outputs/007_rollout/commercial_facility/challenger_sweeps/latest/`
  - Stage-7 can now replay a sweep-derived portfolio winner as a standalone rollout run
    with `resolution=mixed`, `feature_set=portfolio`, `model_label=cross_candidate_portfolio`,
    plus `portfolio_policy_candidate.json` and `shared_origins.csv`
- Stage-7 rollout origin selection is now explicit through `origin_policy`, and the
  challenger sweep now honors the requested policy instead of fanning out across
  unrelated origin-policy audits.
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
- Superseding short-horizon update from the latest focused rerun:
  - challenger recommendation:
    `outputs/007_rollout/commercial_facility/challenger_sweeps/20260311T152035593751Z/recommended_candidate.json`
  - objective: `next_lock_mae` under `origin_policy=phase_balanced`
  - evaluation scope: `origin_selection_scope=shared_timestamp_intersection`
  - learned winner `1min/minimal_phase/hgb-balanced::phase_bucket_next_lock_policy`
  - learned next-lock / path / phase-average MAE:
    `266.837858` (`9.333856%`) / `266.837858` (`9.333856%`) /
    `167.930779` (`5.874136%`)
  - persistence next-lock MAE: `434.846944` (`15.210731%`)
  - best baseline next-lock MAE: `avg_workday` at `419.789637` (`14.684034%`)
- Superseding hourly update from the latest focused rerun:
  - challenger recommendation:
    `outputs/007_rollout/commercial_facility/challenger_sweeps/20260311T011357499505Z/recommended_candidate.json`
  - objective: `next_lock_mae` under `origin_policy=phase_balanced`
  - evaluation scope: `origin_selection_scope=shared_timestamp_intersection`
  - learned winner:
    `cross_candidate_portfolio::phase_bucket_next_lock_policy`
  - learned next-lock / path / profile-shape MAE:
    `253.104260` (`15.969845%`) / `496.893660` (`24.252664%`) /
    `256.446567` (`12.562545%`)
  - persistence next-lock MAE: `379.116458` (`16.733055%`)
  - best baseline next-lock MAE: `hybrid_workday` at `375.929657` (`16.483496%`)
  - interpretation: the current `60m` winner is no longer a single rollout model.
    The strongest measured correction policy is a sweep-derived phase-bucket gate that
    routes bucket `0` to `5min/minimal/hgb-frontier-lr010-l2001::raw` and buckets
    `300`/`600` to `10min/minimal/hgb-balanced::raw`.
- Operational Stage-7 replay of that same `60m` portfolio winner:
  - run:
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
  - interpretation: the repo no longer leaves the best measured `60m` policy stranded
    in sweep artifacts; the same shared-origin gate now exists as a normal Stage-7 run
- Historical secondary `15m` billing-aligned audit (kept for reference, not the
  current operating policy):
  - run:
    `outputs/007_rollout/commercial_facility/20260310T085909851743Z/rollout_selection_summary.csv`
  - learned winner:
    `5min/full/hgb-frontier-lr010-leaf100::anchored_workday_residual`
  - phase-average MAE `315.128523` (`14.477497%`)
  - supporting endpoint/path MAE `373.023169` / `411.357787`
  - persistence in the same run: phase-average MAE `556.036284` (`25.545176%`)
  - interpretation: the strongest current `15m` business-facing model is no longer the
    simple `1min` persistence blend; it is a `5min` anchored-workday residual correction.
- Latest validated boundary-conditioned fallback for that same `15m` learned family:
  - run:
    `outputs/007_rollout/commercial_facility/20260310T085909851743Z/rollout_selection_summary.csv`
  - learned fallback:
    `5min/full/hgb-frontier-lr010-leaf100::hybrid_phase_gate`
  - endpoint/path MAE `267.955348` / `371.917028`
  - phase-average MAE `339.183370` (`15.582614%`)
  - interpretation: the new `hybrid_workday` gate is not the best quarter-hour
    billing objective, but it is now the strongest endpoint/path learned fallback
    for the same short-horizon candidate family.
- Latest validated `15m` phase-balanced audit now promotes derived phase-bucket policy
  candidates that outperform both the prior learned gate and the best baselines:
  - run:
    `outputs/007_rollout/commercial_facility/20260310T085908991319Z/rollout_selection_summary.csv`
  - policy metadata:
    `outputs/007_rollout/commercial_facility/20260310T085908991319Z/rollout_policy_candidates.json`
  - endpoint winner:
    `hgb-frontier-lr010-leaf100::phase_bucket_endpoint_policy`
    at `355.164669` (`15.738267%`)
  - path winner:
    `hgb-frontier-lr010-leaf100::phase_bucket_path_policy`
    at `308.622726` (`14.051080%`)
  - phase-average winner:
    `hgb-frontier-lr010-leaf100::phase_bucket_phase_policy`
    at `175.758571` (`8.001996%`)
  - main comparisons in the same run:
    - prior learned gate `hgb-frontier-lr010-leaf100::hybrid_phase_gate` at
      `439.273838` / `410.067404` / `305.539103`
    - best baselines `persistence` / `persistence` / `hybrid_workday` at
      `444.904222` / `389.439463` / `315.946407`
  - interpretation: the remaining short-horizon robustness gap is no longer "learned
    model vs baseline." The best measured broader-phase result is now a low-cardinality,
    auditable phase-bucket policy built from the evaluated 15-minute candidate surface.

## Stage-8 Horizon Curve Snapshot (2026-03-11)

- Superseding focused H5 rerun:
  - artifact root:
    `outputs/009_horizon_curve/commercial_facility/20260312T065037307284Z/`
  - `1m` learned superiority is not supported on the Stage-5 holdout anchor
  - `15m` and `60m` now resolve to `selection_target=next_lock_mae` under
    `origin_policy=phase_balanced`
  - `1440m` now resolves to `selection_target=profile_shape_mae` under
    `origin_policy=uniform`
  - reused Stage-7 rows now prefer `origin_selection_scope=shared_timestamp_intersection`
    so older pre-fix sweeps do not outrank comparable post-fix runs
  - `15m`: learned next-lock MAE `266.837858` (`9.333856%`) beats persistence
    `434.846944` (`15.210731%`) and best baseline `419.789637` (`14.684034%`)
  - `60m`: learned next-lock MAE `253.104260` (`15.969845%`) from
    `cross_candidate_portfolio::phase_bucket_next_lock_policy` beats persistence
    `379.116458` (`16.733055%`) and best baseline `375.929657` (`16.483496%`)
  - `1440m`: learned profile-shape MAE `717.777613` (`36.245099%`) beats persistence
    `746.527115` (`37.696842%`) while path MAE `783.077104` stays ahead of
    `avg_workday` `850.145715`

- Latest targeted horizon-curve run:
  - `python run_pipeline.py --stage horizon_curve`
  - pipeline: `4.44s`
- H5 artifact root:
  - `outputs/009_horizon_curve/commercial_facility/latest/`
- Method:
  - `1m` uses the Stage-5 promoted holdout anchor
  - `15m` and `60m` use Stage-7 challenger sweeps under `origin_policy=phase_balanced`
    with `selection_target=next_lock_mae`
  - `1440m` uses the Stage-7 challenger sweep under `origin_policy=uniform`
    with `selection_target=profile_shape_mae`
  - matching Stage-7 sweeps are reused from `outputs/007_rollout/commercial_facility/challenger_sweep_registry.csv`; only origin-policy mismatches are rerun
  - reused sweep rows prefer `origin_selection_scope=shared_timestamp_intersection`
- Current objective-aware outcome versus persistence:
  - `1m`: learned superiority is not supported on the current holdout slice
    `174.891813` (`8.436833%`) vs `173.724099` (`8.380502%`)
  - `15m`: learned next-lock MAE `266.837858` (`9.333856%`) beats persistence
    `434.846944` (`15.210731%`) and best baseline `419.789637` (`14.684034%`)
  - `60m`: learned next-lock MAE `253.104260` (`15.969845%`) from
    `cross_candidate_portfolio::phase_bucket_next_lock_policy` beats persistence
    `379.116458` (`16.733055%`) and best baseline `375.929657` (`16.483496%`)
  - `1440m`: learned profile-shape MAE `717.777613` (`36.245099%`) beats persistence
    `746.527115` (`37.696842%`) while path MAE `783.077104` stays ahead of
    `avg_workday` `850.145715`
- Strongest current long-horizon result:
  - `1440m`: learned candidate `10min/minimal/hgb-balanced`
  - learned endpoint/path MAE `968.909580` / `783.077104`
  - `avg_workday` endpoint/path MAE `986.676302` / `850.145715`
  - persistence endpoint/path MAE `1119.137272` / `1010.620668`
- Clear current weak horizons:
  - `15m`: learned endpoint/path MAE `723.780238` / `422.832698`
    vs persistence `577.159993` / `434.852848`
  - `120m`: learned endpoint/path MAE `579.911901` / `607.220358`
    vs persistence `506.931988` / `617.943901`
- Interpretation:
  - H5 is no longer preliminary
  - the repo now exposes a non-monotonic capability envelope rather than a single
    monotonic crossover curve
  - next model work should target the `1m` holdout gap plus endpoint stability at
    `15m` and `120m`, not the previously stale `240m` path gap

## Stage-10 Forecast-Control Snapshot (2026-03-19)

- Latest targeted forecast-control run:
  - `python scripts/modeling/forecast_control_backtest.py`
- Forecast-control artifact root:
  - `outputs/010_forecast_control/commercial_facility/20260322T030301040853Z/`
- Methodology note:
  - this bundle is current under the stricter Stage-10 method:
    - held-out layer promotion
    - real transition-mismatch refresh triggers
    - denser exact-control `1m` blends
    - exact-origin replay reuse plus in-process Stage-7 runtime reuse
    - rolling benchmark cycle catalog, layer inference, and evidence index output
    - full out-of-sample validate/test control replay on the configured schedules
- Control policy used in the backtest:
  - Stage-10 now benchmarks replayed layer candidates on the exact control cycles and
    may promote a stronger baseline over the upstream learned challenger
  - day-ahead: `10min/minimal/hgb-balanced::raw`
  - hourly: `10min/minimal/hybrid_workday` (selected over
    `cross_candidate_portfolio::phase_bucket_next_lock_policy`)
  - isolated phase benchmark winner:
    `1min/minimal_phase_anchor/hgb-balanced::persistence_raw_blend_e25`
    (selected over `1min/minimal_phase/hgb-balanced::phase_bucket_next_lock_policy`)
  - exact stack-selected phase candidate:
    `phase_bucket_portfolio::stack_origin_metric_policy`
  - final applied phase policy: `hourly_passthrough` via
    `10min/minimal/hybrid_workday`, because the broader rolling-support guard
    did not justify a distinct phase correction
  - `1m` nowcast:
    `curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02`
  - Stage-5 holdout still does not support a blanket learned-superiority claim
    at `1m`, but the exact-control Stage-10 minute surface now promotes a
    learned XGBoost control blend over persistence by a large
    operational margin
- Control-cycle coverage:
  - exact-control calibration cycles: `8`
  - exact-control evaluation cycles: `8`
  - rolling calibration cycles: `16`
  - rolling evaluation cycles: `16`
  - train remains excluded from promoted quality claims; this is now the full
    eligible validate/test control surface on the configured schedules
- Aggregate locked-interval and profile results:
  - frozen day-ahead lock MAE: `767.411283` (`40.518170%`)
  - after hourly updates lock MAE: `490.428482` (`25.160719%`)
  - after phase updates lock MAE: `490.428482` (`25.160719%`)
  - after nowcast updates lock MAE: `47.503499` (`2.459006%`)
  - frozen day-ahead profile-shape MAE: `788.533702` (`41.185297%`)
  - after hourly updates profile-shape MAE: `626.681554` (`32.175992%`)
  - after phase updates profile-shape MAE: `626.681554` (`32.175992%`)
  - after nowcast updates profile-shape MAE: `174.956343` (`9.038661%`)
  - frozen day-ahead energy MAE: `443965.266129` (`15.740564%`)
  - after phase updates energy MAE: `65667.574693` (`2.484910%`)
  - after nowcast updates energy MAE: `632.960874` (`0.023103%`)
- Rolling benchmark:
  - calibration cycles: `16`
  - evaluation cycles: `16`
  - rolling evaluation lock MAE:
    `763.962699 -> 492.201440 -> 492.201440 -> 47.500033`
  - rolling evaluation profile-shape MAE:
    `786.255244 -> 626.787911 -> 626.787911 -> 175.213594`
  - rolling hourly-vs-day-ahead lock gain:
    `271.761259`, 95% CI [`146.509717`, `379.601589`], `p=0.0000`
  - rolling hourly-vs-day-ahead profile gain:
    `159.467333`, 95% CI [`80.165916`, `228.327102`], `p=0.0000`
  - rolling phase-vs-hourly lock gain:
    `0.000000`, 95% CI [`0.000000`, `0.000000`], `p=1.0000`
  - rolling nowcast-vs-phase lock gain:
    `444.701407`, 95% CI [`408.156351`, `482.162622`], `p=0.0000`
- Day-ahead refresh study:
  - refresh candidate: `10min/minimal/hgb-balanced::hybrid_workday_residual`
  - frozen day-ahead lock MAE: `767.411283` (`40.518170%`)
  - unconditional refresh lock MAE: `606.603723` (`31.678078%`)
  - triggered refresh lock MAE: `655.385169` (`33.752357%`)
  - frozen day-ahead profile-shape MAE: `788.533702` (`41.185297%`)
  - unconditional refresh profile-shape MAE: `701.862380` (`36.420548%`)
  - triggered refresh profile-shape MAE: `732.516445` (`37.704136%`)
  - mean triggered refresh updates applied per cycle: `8.750`
  - trigger rate on the exact control cycles: `0.3804347826`
  - rolling trigger rate: `0.3838028169`
  - selected trigger mode: `residual_or_activity_active_or_transition`
  - triggered refresh preserved `64.63%` of the unconditional
    profile-shape gain and `69.66%` of the unconditional lock-MAE gain
  - current recommended operating mode is therefore `triggered_refresh`
  - current trigger reason mix now includes both residual drift and activity-profile shift
  - rolling benchmark also recommends `triggered_refresh`:
    `649.485955` / `730.181648` for triggered
- Focused standalone `1440m` challenger sweep:
  - `outputs/007_rollout/commercial_facility/challenger_sweeps/20260320T090013545419Z/`
  - standalone winner remains `10min/minimal/hgb-balanced::raw`
  - profile-shape MAE `717.777613` (`36.245099%`)
  - `hybrid_workday_residual` remains more useful as a refresh path than as the
    main frozen `24h` rollout
  - interpretation: the residual model is useful as an intraday refresh path,
    not yet as the main frozen day-ahead anchor
- Per-cycle readout:
  - `control_backtest_by_cycle.csv` contains the persisted cycle-level exact
    replay trace for the latest bundle
- Cache and reproducibility:
  - exact-origin replay cache registry:
    `outputs/010_forecast_control/commercial_facility/replay_cache/replay_cache_registry.csv`
  - current evidence index:
    `outputs/010_forecast_control/commercial_facility/latest/current_evidence_index.md`
- Interpretation:
  - the repo still has direct end-to-end evidence that intraday updates improve the
    frozen day-ahead forecast on this test window
  - the repo now exposes a fully persisted four-layer control stack under the
    stricter methodology and all-eligible out-of-sample validate/test replay
  - the new day-ahead refresh study shows that the repo can improve the frozen
    `24h` profile with a learned residual refresh, even though the same
    residual candidate does not yet win as the best standalone `1440m` rollout
  - the current operational control stack is learned at day-ahead, learned at
    hourly, hourly passthrough at the structural `15m` phase slot, and learned
    at `1m` on the exact-control surface
  - the broader rolling benchmark now strengthens the hourly result and is also
    the reason the distinct exact-stack phase candidate is not currently applied
  - the `1m` control benchmark now uses the Stage-5 holdout registry, latest
    `blend_finalists.csv`, and the remaining widened short-horizon scoreboard, and
    replays learned `+blend` candidates with their saved Stage-5 blend settings
  - the next questions are now narrower:
    - whether the refresh trigger can capture more of the unconditional-refresh gain
    - whether a learned `1m` anchor can beat persistence on the honest Stage-5 holdout
    - whether current-state documentation continues to stay synced with artifact-backed evidence

> **Note:** The figures below are generated by running Stage-10 and are not
> stored in version control. Run
> `python scripts/modeling/forecast_control_backtest.py` to produce them locally.

![Stage-10 locked-interval MAE progression](../../outputs/010_forecast_control/commercial_facility/latest/fig_control_lock_mae.png)

![Stage-10 example control cycle](../../outputs/010_forecast_control/commercial_facility/latest/fig_control_example_cycle.png)

![Stage-10 day-ahead refresh policy comparison](../../outputs/010_forecast_control/commercial_facility/latest/fig_day_ahead_refresh_policy.png)

## Spec Alignment Note

The canonical H1 control label in executed notebook artifacts is `temporal_no_workday`.
Repository docs/specs were aligned to this exact label to remove naming drift.

## Contribution Attribution

Implementation ownership for this run should be read as:

- Team (Spencer, Sean, Frank): primary project design and execution baseline (~70%)
- AI QA hardening/support: gap detection, consistency checks, and documentation alignment (~30%)

This split reflects project reality and preserves team-first credit in report framing.
