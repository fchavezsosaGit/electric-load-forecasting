# MVMP (Minimum Viable Modeling Product)

> Historical note:
> This document preserves the notebook-era `1min` MVP anchor. It is still useful
> for understanding how the repo started, but it is not the active operating
> roadmap. The active direction now lives in
> [002_operating_direction_spec.md](../000_governance/002_operating_direction_spec.md)
> and [operational_hypotheses.md](operational_hypotheses.md).

This document defines the first constrained modeling target for the electric load
forecasting project. The MVMP is intentionally narrow: it verifies that the full
pipeline (data ingestion through evaluation) works end-to-end before the team invests
in more complex models or feature engineering.

Related references:
- [000_spec.md](../000_governance/000_spec.md)
- [002_operating_direction_spec.md](../000_governance/002_operating_direction_spec.md)
- [feature_sets.md](feature_sets.md)
- [hypothesis.md](hypothesis.md)
- [glossary.md](../004_reference/glossary.md)

## Scope

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Resolution | `1min` | Matches Report IV MVP execution and preserves short-horizon nowcasting behavior |
| Feature set anchor | `minimal` | 3 predictors (`workday`, `hour`, `lag_1`) for a controlled starting point |
| Model families | Ridge + HistGradientBoostingRegressor | Regularized linear baseline plus fixed nonlinear comparators |
| Primary metric | MAE | Directly interpretable in watts; measures average prediction error |
| Secondary metric | RMSE | Penalizes large errors; useful for detecting poor performance during load transitions |

## Why this scope

- The current Report IV decision is to lock MVP execution to `1min` so hypothesis
  evidence is based on one consistent resolution.
- `minimal` is still the anchor for controlled comparisons, but the executed notebook
  evaluates all five canonical feature sets (`minimal`, `temporal`, `curated`, `full`,
  `full_stable`).
- The active `temporal`, `curated`, `full`, and `full_stable` sets now include
  continuous daily/weekly Fourier encodings so cyclical structure is represented
  smoothly for nonlinear models.
- Ridge provides a deterministic regularized linear baseline; HGB adds nonlinear
  behavior checks without changing data contracts.
- Multi-resolution and recursive-rollout support now exist as Stage-6 and Stage-7
  scripted workflows, but they remain outside the canonical MVP notebook so Report-IV
  claims stay anchored to the `1min` scope.

## Dataset sizes (approximate, `1min`)

| Split | Days | Rows (1min) | Date Range |
|-------|------|-------------|------------|
| Train | 25 | ~36,000 | Nov 28 - Dec 22 |
| Validate | 3 | ~4,320 | Dec 23 - Dec 25 |
| Test | 3 | ~4,320 | Dec 26 - Dec 28 |

Note: Exact row counts depend on gold-layer null filtering. December 25 (Christmas)
falls in the validation split, which may affect representativeness for business-day
predictions. This is documented for awareness but not altered given the 31-day dataset.

## Success threshold (first pass)

The MVMP is successful when all conditions below are met:

1. **Reproducibility**: Running the pipeline and model training twice with the same
   inputs produces consistent metrics and artifact files (`run_manifest.json`,
   `metrics_overall.csv`).
2. **Protocol integrity**: Hypotheses are evaluated on validation only, and holdout test
   is executed once after model selection.
3. **Baseline transparency**: Performance versus persistence/previous-day/avg-workday is
   explicitly reported, regardless of whether a model beats persistence.
4. **Artifact completeness**: `outputs/004_modeling/<artifact_namespace>/` contains
   required CSV/PNG files, including normalized `mae_pct` / `rmse_pct` fields.

## Persistence baseline definition

The persistence baseline predicts that the load at time t equals the load at time t-1:

```text
y_hat(t) = y(t-1)
```

This is equivalent to using `lag_1` as the sole predictor with a coefficient of 1.0 and
an intercept of 0.0. Any model that cannot beat this baseline is not capturing useful
signal from the feature set.

## Hypothesis mapping

| Hypothesis | Resolution | Feature sets compared | Model family | Primary metric |
|-----------|------------|----------------------|--------------|----------------|
| H1 (workday signal) | `1min` | minimal vs temporal-minus-workday control | Ridge (primary), HGB (cross-check) | MAE |
| H2 (lag/transition value) | `1min` | temporal vs curated | Ridge and HGB | RMSE |
| H3 (resolution tradeoff) | multi-resolution | matched-horizon Stage-6 grid | Ridge/HGB + persistence | MAE |
| H4 (exploratory nonlinear behavior) | `1min` | all feature sets | Ridge vs HGB | MAE and RMSE |
| H5 (horizon degradation) | multi-horizon | selected model at increasing lead times | Selected model + persistence | MAE ratio |

See [hypothesis.md](hypothesis.md) for full hypothesis definitions and experimental
designs.

## What comes after MVMP

Once the MVMP criteria are met, the notebook-era scope is considered complete.
Further work should be driven by the active layered operating spec instead of by
re-expanding this MVP:

- keep MVMP as provenance for the original `1min` notebook question
- use Stage-5 through Stage-10 and the active operating spec for current model
  and delivery decisions
- do not use this document alone to justify new `1m`-first claims or a
  single-model roadmap
