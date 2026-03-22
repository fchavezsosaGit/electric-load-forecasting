# Model and Blend Guide

This guide explains how to read the repo's candidate labels, what the blend
wrappers mean, why different stages can have different winners, and how the
current CPU/GPU policy fits into that story.

The repo is intentionally honest here: if a learned model loses to a baseline,
the docs should say so plainly. A baseline win is still useful evidence.

## Start Here

- If the question is "what is the deployable `1m` anchor right now?", trust
  Stage-5. The current honest answer is still `persistence`.
- If the question is "which horizon-specific policy helps on its own
  objective?", trust Stage-8. That is the capability-envelope surface.
- If the question is "what stack should an optimizer consume for day-ahead plus
  intraday control?", trust Stage-10. That is the exact-control replay surface.
- If two stages disagree, that is not automatically a bug. They are often
  answering different operational questions on different replay surfaces.

## Label Anatomy

Stage-5 labels use this shape because Stage-5 is fixed to `1min`:

```text
<feature_set>/<model_label>/<target_mode>
```

Examples:

- `curated_ramp/hgb-auto-lr050-d7-leaf100-l20000-it300/residual+blend`
- `full_stable/hgb-frontier-lr010-l2001/raw+blend`

Stage-7, Stage-8, and Stage-10 labels include resolution explicitly:

```text
<resolution>/<feature_set>/<model_label>::<target_mode>
```

Examples:

- `10min/minimal/hgb-balanced::raw`
- `5min/full/hgb-frontier-lr010-leaf100::anchored_workday_residual`

Special case:

- `mixed/portfolio/cross_candidate_portfolio::...` means the repo did not keep
  one single source candidate. It built a measured routing policy from multiple
  candidates on shared origins.

Wrapper separators matter:

- `/` separates Stage-5 feature set, model label, and target mode.
- `::` separates an upstream rollout/control candidate from its target mode.
- `+` denotes a Stage-5 short-horizon blend wrapper layered onto a base target
  mode.
- `|` denotes a Stage-10 control-stack wrapper layered onto an upstream
  candidate.

## Model Labels

- `ridge-*`: regularized linear baselines used for transparent, low-capacity
  comparisons.
- `hgb-balanced`: the conservative HistGradientBoostingRegressor default.
- `hgb-frontier-*`: manually chosen higher-capacity HGB variants used in the
  broader frontier.
- `hgb-auto-*`: bounded adaptive HGB variants promoted from the centralized
  Stage-5 search, not ad hoc one-off labels.
- `xgb-*`: optional XGBoost candidates. They are only eligible when the
  acceleration extra is installed and the runtime probe confirms a usable
  device.
- `cross_candidate_portfolio`: a sweep-derived policy that routes different
  phase buckets to different measured source candidates.

The current practical takeaway is simple:

- the repo can use optional GPU-backed XGBoost on capable x64/NVIDIA hosts
- the current operational winners are still HGB-based
- GPU is therefore available where it helps, but it is not forced when the
  measured winner is CPU/HGB

## Target Modes And Blend Wrappers

Base target modes:

- `raw`: predict load directly.
- `residual`: predict a correction relative to a baseline and add that baseline
  back at inference time.
- `anchored_workday_residual`: predict a correction on top of the
  anchored-workday baseline path.
- `hybrid_workday_residual`: predict a correction on top of the hybrid-workday
  baseline path.

Stage-5 wrappers:

- `raw+blend`: direct model plus the Stage-5 causal sigmoid blend guardrail.
- `residual+blend`: residual model plus the same sigmoid blend guardrail.
- `...+bucket_blend_b5`: Stage-5 bucketized blend with separate weights for
  5-minute buckets inside the short correction window.
- `...+blend_bucket_blend_b5`: a nested Stage-5 wrapper that applies the
  sigmoid blend first and then a bucketized blend on top of it.

Operational meaning:

- Stage-5 blend wrappers are short-horizon guardrails. They allow the learned
  candidate to stay close to persistence or another safer reference when recent
  evidence says the learned path is in a weaker patch.

Stage-7 and Stage-10 policy-style target modes:

- `hybrid_phase_gate`: a phase-conditioned gate that blends toward the hybrid
  workday fallback differently across the quarter-hour phase.
- `phase_bucket_next_lock_policy`: a low-cardinality phase-bucket routing
  policy tuned for the next lock interval objective.
- `baseline_control_bucket_blend_b5`: a Stage-10 stack wrapper that blends a
  learned `15m` phase candidate toward the best reconstructable baseline with
  5-minute bucket weights inside the control window.
- `control_bucket_blend_b5`: the analogous Stage-10 wrapper for the final
  minute nowcast layer.
- `control_blend_w0.02`: a Stage-10 scalar control-surface blend whose suffix
  records the calibrated control-layer blend weight chosen on held-out replay.

Example decodes:

- `phase_bucket_portfolio::stack_origin_metric_policy`
  means:
  - this is not one upstream phase candidate carried through unchanged
  - Stage-10 built a stack-aware portfolio that routes phase corrections using
    the origin/phase-bucket control evidence instead of keeping a single source
    candidate everywhere
- `curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02`
  means:
  - base candidate: `curated_ramp/xgb-balanced/residual+blend`
  - stack wrapper: Stage-10 then applies a calibrated low-weight control blend
    against its safer reference policy before promoting it as the final minute
    layer

## How Candidates Are Created

Stage-5 (`1m` holdout gate):

- trains a fixed `1min` short-horizon grid
- compares feature sets and model families
- evaluates both `raw` and `residual` variants when allowed
- optionally searches blend wrappers
- optionally adds XGBoost only on accelerated hosts
- promotes one learned challenger only after coverage and fold-quality guards
- still keeps `persistence` as the deployment recommendation unless the learned
  challenger beats the strongest baseline on holdout

Stage-7 and Stage-8 (rollout and horizon characterization):

- replay recursive candidates on shared origins
- rank them on the metric that matters for that horizon
- allow policy candidates such as phase-bucket routing or cross-candidate
  portfolios when the evidence supports them
- characterize horizon capability, not just a single global leaderboard

Stage-10 (control-stack backtest):

- does not blindly trust upstream winners
- replays candidate pools on the exact control cycles
- can swap an upstream winner for a different operational layer if the replayed
  control surface says the swap is better

That last rule is why the Stage-10 `1m` nowcast winner can differ from the
latest Stage-5 holdout promotion candidate without either result being wrong.

## Current Honest Readout

Current `1m` anchor truth:

- Stage-5 deployment recommendation: `persistence`
- Stage-5 operating policy artifact:
  `outputs/005_performance/<artifact_namespace>/latest/operating_policy.json`
- best current learned Stage-5 challenger:
  `curated_ramp/xgb-balanced/residual+blend`
- honest interpretation: the learned challenger is credible enough to keep
  studying, but not credible enough to replace persistence as the standalone
  `1m` deployment anchor yet; the current holdout coverage summary also says the
  reviewed standalone surface is narrow-regime evidence, and the repo now writes
  that distinction explicitly instead of leaving it implicit in the holdout table
- Stage-5 supplemental advisory artifact:
  `outputs/005_performance/<artifact_namespace>/latest/supplemental_surface_advisory.json`
- broader advisory interpretation: the stitched validate-walkforward plus
  holdout surface is learned-positive overall and specifically supports learned
  `1m` corrections in `transition_only` and `transition_active`, but that does
  not override the canonical standalone holdout gate

Current horizon-curve truth:

- Stage-8 `1m` row:
  `curated_ramp/xgb-balanced/residual+blend`
- honest interpretation: that is the current persisted horizon-curve
  characterization point, not the deployable Stage-5 recommendation

Current optimizer-facing control truth:

- day-ahead: `10min/minimal/hgb-balanced::raw`
- hourly: `10min/minimal/hybrid_workday`
- exact stack-selected phase candidate:
  `phase_bucket_portfolio::stack_origin_metric_policy`
- currently applied phase slot: `10min/minimal/hybrid_workday`
  via `hourly_passthrough`
- minute nowcast:
  `curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02`

Honest interpretation:

- the repo still does not support a blanket learned-superiority claim at `1m`
- Stage-5 now says that explicitly as a baseline-led standalone policy, while
  still leaving room for learned `1m` corrective overlays downstream
- the repo does support a strong stacked forecast/nowcast/control story for the
  optimizer-facing surface
- the distinct exact-stack phase candidate is still useful evidence, but the
  current broader rolling-support guard keeps the applied phase slot on hourly
  passthrough rather than claiming an independent `15m` win
- the current best operational `1m` layer is a learned XGBoost control blend on
  the accelerated host, while CPU-safe fallback paths remain the portable
  default contract
- dynamic minute routing is still shadow-only; the current persisted
  counterfactual says hard enforcement would hurt all-interval error
- GPU-backed XGBoost is available because the replayed control surface currently
  prefers it, not because the repo is forcing GPU for its own sake

## Compute Policy

- optional acceleration is enabled through the acceleration extra and runtime
  device probe
- ARM64 or non-accelerated teammate machines stay on the CPU-safe path
- capable x64/NVIDIA hosts can evaluate GPU-backed XGBoost challengers
- current operational winners should still be chosen by measured quality, not
  by forcing GPU usage

The current measured result is:

- GPU is available and used where eligible
- HGB still owns the day-ahead and most coarse-horizon operating layers
- the latest exact-control `1m` winner is a sparse-feature HGB control-bucket
  blend, not an XGBoost overlay

## Suggested Reading Order

1. [current_validation_snapshot.md](current_validation_snapshot.md)
2. [current_operating_approach.md](current_operating_approach.md)
3. [`../../outputs/005_performance/commercial_facility/latest/promotion_candidate.json`](../../outputs/005_performance/commercial_facility/latest/promotion_candidate.json)
4. [`../../outputs/009_horizon_curve/commercial_facility/latest/horizon_curve_summary.md`](../../outputs/009_horizon_curve/commercial_facility/latest/horizon_curve_summary.md)
5. [`../../outputs/010_forecast_control/commercial_facility/latest/control_policy.json`](../../outputs/010_forecast_control/commercial_facility/latest/control_policy.json)
