# Feature Sets

This document defines canonical modeling feature sets derived from gold data. Each set
is a named, versioned subset of predictor columns used for model training and evaluation.

Source of truth for column lists is `scripts/config.py` (`FEATURE_SETS`).

Related references:
- [000_spec.md](../000_governance/000_spec.md)
- [hypothesis.md](hypothesis.md)
- [mvmp.md](mvmp.md)
- [glossary.md](../004_reference/glossary.md)

## Overview

| Canonical name | Legacy label | Column count | Primary purpose |
|----------------|-------------|-------------|-----------------|
| `minimal` | A | 3 | Fast baseline with minimal context |
| `temporal` | B | 10 | Calendar structure + immediate lag |
| `full` | C | 41 | Maximum information content benchmark |
| `curated` | D | 11 | Balanced signal with reduced collinearity |

## Set Definitions

### `minimal` (A)

| Column | Category | Description |
|--------|----------|-------------|
| `workday` | Business | Ternary day-type encoding (0/1/2) |
| `hour` | Temporal | Hour of day (0-23) |
| `lag_1` | Lag | Previous-period load value |

Rationale:
- Smallest possible set that combines business context, time-of-day, and recent load
  history.
- Used as the MVMP feature set and as a baseline for measuring the marginal value of
  additional features.
- Tests whether immediate lag + coarse time/business context is enough to beat a naive
  persistence baseline.

Risks:
- Likely to underfit during sudden load ramps and holiday behavior where more context
  is needed.

Hypothesis connection:
- H1 uses `minimal` as treatment to test the value of `workday`.
- H3 uses `minimal` across resolutions to isolate the resolution effect.

### `temporal` (B)

| Column | Category | Description |
|--------|----------|-------------|
| `workday` | Business | Ternary day-type encoding (0/1/2) |
| `year` | Temporal | Calendar year |
| `quarter` | Temporal | Quarter of year (1-4) |
| `month` | Temporal | Month (1-12) |
| `day` | Temporal | Day of month (1-31) |
| `day_of_week` | Temporal | Day of week (0=Sunday through 6=Saturday) |
| `hour` | Temporal | Hour of day (0-23) |
| `season` | Temporal | Meteorological season (1-4) |
| `time_of_day` | Temporal | Coarse time bucket (0-3) |
| `lag_1` | Lag | Previous-period load value |

Rationale:
- Adds full calendar structure while keeping feature count moderate (10 columns).
- Tests whether temporal context beyond `hour` improves predictions over the minimal set.
- Serves as the control condition for hypothesis H2 (lag value).

Risks:
- Calendar features (`year`, `quarter`, `month`) can be weak predictors on a 31-day
  dataset where there is minimal variation in those values.
- Some temporal features are partially redundant (e.g., `season` is determined by `month`).

### `full` (C)

Columns:
- All 44 gold columns except `timestamp`, `day_class`, and `avg_load` (target).
- Total: 41 predictor columns.

Rationale:
- Highest information content and strongest raw predictive potential.
- Benchmark for best-case accuracy before feature pruning.
- Useful for identifying which features contribute the most through coefficient analysis
  or importance rankings.

Risks:
- High collinearity between lag, rolling, and delta features (EDA shows many pairs with
  |r| > 0.95). This can destabilize linear model coefficients and inflate variance.
- Heavier training cost and slower iteration compared to smaller sets.

### `curated` (D)

| Column | Category | Description |
|--------|----------|-------------|
| `workday` | Business | Ternary day-type encoding (0/1/2) |
| `hour` | Temporal | Hour of day (0-23) |
| `season` | Temporal | Meteorological season (1-4) |
| `time_of_day` | Temporal | Coarse time bucket (0-3) |
| `lag_1` | Lag | Previous-period load value (short memory) |
| `lag_5` | Lag | 5-period load value (medium-short memory) |
| `lag_60` | Lag | 60-period load value (hourly memory) |
| `lag_1440` | Lag | 1440-period load value (daily cycle memory) |
| `rolling_mean_15` | Rolling | 15-period trailing mean (short smoothing) |
| `rolling_std_60` | Rolling | 60-period trailing std (volatility indicator) |
| `slope_15` | Slope | 15-period linear trend (direction indicator) |

Rationale:
- Reduced set that balances signal, interpretability, and training speed.
- Removes redundant rolling/lag columns (identified via silver EDA correlation analysis)
  while keeping short-term, medium-term, and daily-cycle memory.
- Includes volatility (`rolling_std_60`) and trend direction (`slope_15`) features that
  are not available in `minimal` or `temporal`.
- Serves as the treatment condition for hypothesis H2 (lag value).

Risks:
- May omit interaction effects that matter for edge cases (e.g., seasonal load ramps
  that depend on specific rolling window sizes not included).
- `lag_1440` introduces warm-up NaN for the first 1440 periods, which reduces effective
  training data at the start of the series.

## Target Column

- `avg_load` is the prediction target and is included separately in all model dataset
  files. It is explicitly excluded from every feature set to prevent target leakage.
- This exclusion is enforced at runtime by `scripts/003_create_model_datasets.py`.

## Consistency Rule

Before training any model, verify that all feature set columns exist in the gold data:

```python
set(FEATURE_SETS[set_name]) - set(gold_df.columns) == set()
```

If the result is non-empty, either the pipeline schema or this document must be updated
before proceeding.

## Resolution Note

These feature-set definitions are resolution-agnostic. The same column names apply to
all supported resolutions (`1s` through `15min`) produced by the pipeline. Lag and
rolling window sizes are in periods (not fixed time), so their real-world duration
scales with resolution. For example, `lag_60` at `5min` resolution looks back 300
minutes (5 hours), while at `1min` resolution it looks back only 60 minutes (1 hour).

