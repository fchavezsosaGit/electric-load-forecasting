# MVMP (Minimum Viable Modeling Product)

This document defines the first constrained modeling target for the electric load
forecasting project. The MVMP is intentionally narrow: it verifies that the full
pipeline (data ingestion through evaluation) works end-to-end before the team invests
in more complex models or feature engineering.

Related references:
- [000_spec.md](../000_governance/000_spec.md)
- [feature_sets.md](feature_sets.md)
- [hypothesis.md](hypothesis.md)
- [glossary.md](../004_reference/glossary.md)

## Scope

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Resolution | `5min` | Reduces dataset to 8,928 rows (5x smaller than `1min`) while preserving daily load shape |
| Feature set | `minimal` | 3 predictors (`workday`, `hour`, `lag_1`) -- tests signal without feature overload |
| Model | Linear Regression | Fast, interpretable, and ideal for validating the end-to-end modeling path |
| Primary metric | MAE | Directly interpretable in watts; measures average prediction error |
| Secondary metric | RMSE | Penalizes large errors; useful for detecting poor performance during load transitions |

## Why this scope

- `5min` reduces dataset size by 5x vs `1min` while preserving daily structure. This
  makes iteration faster during initial development without sacrificing the load dynamics
  needed for meaningful evaluation.
- `minimal` tests whether the pipeline can deliver useful signal with only three features.
  If the minimal set already beats a naive baseline, the pipeline infrastructure is
  validated and the team can iterate on features with confidence.
- Linear Regression is fast, interpretable, and produces a clear baseline against which
  all future models can be compared. It also avoids introducing hyperparameter tuning
  complexity before the pipeline is proven.
- `15min` is retained as a required operational resolution when interval billing applies.
  MVMP remains at `5min` for first-pass model iteration speed.

## Dataset sizes (approximate)

| Split | Days | Rows (5min) | Date Range |
|-------|------|-------------|------------|
| Train | 25 | ~7,200 | Nov 28 - Dec 22 |
| Validate | 3 | ~864 | Dec 23 - Dec 25 |
| Test | 3 | ~864 | Dec 26 - Dec 28 |

Note: Exact row counts depend on gold-layer null filtering. December 25 (Christmas)
falls in the validation split, which may affect representativeness for business-day
predictions. This is documented for awareness but not altered given the 31-day dataset.

## Success threshold (first pass)

The MVMP is successful if all three conditions are met:

1. **Reproducibility**: Running the pipeline and model training twice with the same
   inputs produces identical metrics. No randomness, no data leakage, no manual steps.
2. **Baseline improvement**: The Linear Regression model beats a naive persistence
   baseline (predict next value = most recent observed value) on validation MAE.
3. **Stability**: Metrics do not change across reruns, confirming deterministic behavior
   from data ingestion through evaluation.

## Persistence baseline definition

The persistence baseline predicts that the load at time t equals the load at time t-1:

```text
y_hat(t) = y(t-1)
```

This is equivalent to using `lag_1` as the sole predictor with a coefficient of 1.0 and
an intercept of 0.0. Any model that cannot beat this baseline is not capturing useful
signal from the feature set.

## Hypothesis mapping

| Hypothesis | Resolution | Feature sets compared | Model | Primary metric |
|-----------|------------|----------------------|-------|----------------|
| H1 (workday signal) | 5min | minimal vs temporal | Linear Regression | MAE |
| H2 (lag value) | 5min | temporal vs curated | Linear Regression | RMSE |
| H3 (resolution tradeoff) | 1min vs 5min | minimal | Linear Regression | MAE |

See [hypothesis.md](hypothesis.md) for full hypothesis definitions and experimental
designs.

## What comes after MVMP

Once the MVMP criteria are met, the team can proceed to:
- Testing additional feature sets (`temporal`, `curated`, `full`) against the baseline.
- Evaluating hypothesis H1, H2, and H3 as defined in the hypothesis document.
- Introducing more complex models (tree-based, regularized linear) if the feature set
  experiments warrant it.
- Expanding to additional resolutions (`1min`, `15min`) for operational evaluation.

