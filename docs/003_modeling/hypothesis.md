# Hypotheses

This document defines Report IV hypotheses that connect EDA findings to the executed
modeling notebook (`notebooks/003_modeling.ipynb`).

Format:
EDA shows `[observation]`. We hypothesize that `[approach]` will `[effect]` as measured
by `[metric]` achieving `[target]`.

Related references:
- [000_spec.md](../000_governance/000_spec.md)
- [mvmp.md](mvmp.md)
- [feature_sets.md](feature_sets.md)
- [glossary.md](../004_reference/glossary.md)

## Summary Table

| ID | Name | Metric | Target | Resolution | Status |
|----|------|--------|--------|------------|--------|
| H1 | Workday signal | MAE | >=10% improvement | `1min` | Evaluated |
| H2 | Lag/rolling value | RMSE | >=8% improvement | `1min` | Evaluated |
| H3 | Resolution tradeoff | MAE | <=5% degradation (`5min` vs `1min`) | multi-resolution | Deferred |
| H4 | Nonlinear behavior vs regularized linear baseline | MAE and RMSE | Exploratory | `1min` | Evaluated |

---

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
- Note: RMSE is primary for H2 because it emphasizes larger misses.
- Note: Ridge drops feature-NaN rows during fit, so `curated`/`full` can have fewer
  effective training rows than `minimal`/`temporal`; HGB is included as a cross-check
  because it can train with feature NaNs without that row-drop behavior.

Metric and target:
- Primary metric: RMSE
- Target: >=8% RMSE improvement

## H3: Resolution tradeoff (deferred)

Observation:
Coarser resolutions can reduce noise and training cost, but may lose short-horizon
fidelity.

Hypothesis:
We hypothesize that a `5min` model can achieve MAE within **5%** of an equivalent
`1min` model while using fewer training rows.

Status:
- **Deferred for Report IV MVP**.
- Current notebook executes only `1min` resolution by design.
- H3 will be evaluated when multi-resolution runs (`1min`, `5min`) are re-enabled.

Planned comparison design (future run):
- Same feature set (`minimal`) and same model family across resolutions.
- One-step-ahead evaluation at each native resolution.
- No interpolation/alignment to a shared grid in the MVP path.

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
- Evaluation split: validation
- Additional holdout check: one-shot test after selection

Metric framing:
- Track both MAE and RMSE.
- No hard pass/fail threshold for H4; interpret as comparative behavior analysis.

