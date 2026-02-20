# Hypotheses

This document defines testable hypotheses that connect EDA observations to modeling
experiments. Each hypothesis follows the structured format from DSE 260 Lecture 3:

Format:
EDA shows `[observation]`. We hypothesize that `[approach]` will `[effect]` as measured
by `[metric]` achieving `[target]`.

Related references:
- [000_spec.md](../000_governance/000_spec.md)
- [mvmp.md](mvmp.md)
- [feature_sets.md](feature_sets.md)
- [glossary.md](../004_reference/glossary.md)

## Summary Table

| ID | Name | Metric | Target | Resolution | Feature Sets |
|----|------|--------|--------|------------|-------------|
| H1 | Workday signal | MAE | >=10% improvement | 5min | minimal vs temporal |
| H2 | Lag value | RMSE | >=8% improvement | 5min | temporal vs curated |
| H3 | Resolution tradeoff | MAE | <=5% degradation | 1min vs 5min | minimal |

---

## H1: Workday signal

Observation:
EDA shows clear separation of daily load profiles across `day_class` (`full`, `half`,
`none`). Full working days exhibit higher daytime peaks and steeper morning ramps
compared to non-working days. Half days fall between the two. This separation is
visible in both the raw overlay plots (`notebooks/000_raw_eda.ipynb`) and the silver
workday profile analysis (`notebooks/002_silver_eda.ipynb`).

Hypothesis:
We hypothesize that including `workday` in the feature set will reduce validation MAE
by at least **10%** relative to a temporal-only baseline at the same resolution.

Experimental design:
- Model: Linear Regression
- Resolution: `5min`
- Control: `temporal` feature set (calendar features + `lag_1`, without `workday`)
- Treatment: `minimal` feature set (`workday` + `hour` + `lag_1`)
- Evaluation: Compare MAE on the validation split (days 26-28)
- Note: The `minimal` set has fewer features than `temporal`, so an improvement would
  indicate that `workday` provides stronger signal than the additional calendar columns.

Metric and target:
- Primary metric: MAE
- Target: >=10% MAE improvement

## H2: Short and medium lag value

Observation:
EDA shows strong short-term autocorrelation in `avg_load` at 1-minute resolution
(`notebooks/002_silver_eda.ipynb`, ACF plot). The autocorrelation remains significant
through at least 240 lags (4 hours), and a daily cycle is visible at lag 1440.
This suggests that recent and daily-cycle load values carry predictive information
beyond what calendar features alone provide.

Hypothesis:
We hypothesize that adding lag features (`lag_1`, `lag_5`, `lag_60`, `lag_1440`) to
linear regression will reduce validation RMSE by at least **8%** vs a model with only
temporal and business features.

Experimental design:
- Model: Linear Regression
- Resolution: `5min`
- Control: `temporal` feature set (calendar + business features + `lag_1`)
- Treatment: `curated` feature set (adds `lag_5`, `lag_60`, `lag_1440`, rolling and
  slope features)
- Evaluation: Compare RMSE on the validation split (days 26-28)
- Note: RMSE is chosen as the primary metric here because lag features are expected
  to reduce large errors during load transitions, and RMSE penalizes large errors
  more than MAE does.

Metric and target:
- Primary metric: RMSE
- Target: >=8% RMSE improvement

## H3: Resolution tradeoff

Observation:
EDA shows that 5-minute aggregation reduces high-frequency noise while preserving the
overall daily load shape (`notebooks/002_silver_eda.ipynb`, multi-resolution hourly
profile comparison). The 5-minute dataset has 5x fewer rows (8,928 vs 44,640) than
the 1-minute dataset, which directly reduces training time and memory usage.

Hypothesis:
We hypothesize that a 5-minute model will achieve MAE within **5%** of the equivalent
1-minute model MAE, while training on significantly fewer rows.

Experimental design:
- Model: Linear Regression
- Feature set: `minimal` (held constant to isolate the resolution effect)
- Control: `1min` resolution (44,640 rows before gold filtering)
- Treatment: `5min` resolution (8,928 rows before gold filtering)
- Evaluation: Compare MAE on the test split (days 29-31) at each resolution.
  For fair comparison, both predictions are evaluated against the same ground-truth
  load values, interpolated or aligned to a common resolution if needed.
- Note: If the 5-minute model meets the target, it validates the MVMP choice of `5min`
  as the default modeling resolution.

Metric and target:
- Primary metric: MAE
- Target: 5-minute MAE no worse than +5% vs 1-minute

