# Feature Sets

This document defines canonical modeling feature sets derived from gold data. Each set
is a named, versioned subset of predictor columns used for model training and evaluation.

Source of truth for column lists is `scripts/config.py` (`FEATURE_SETS`).

Related references:
- [000_spec.md](../000_governance/000_spec.md)
- [hypothesis.md](hypothesis.md)
- [model_and_blend_guide.md](model_and_blend_guide.md)
- [mvmp.md](mvmp.md)
- [glossary.md](../004_reference/glossary.md)

## Overview

| Canonical name | Legacy label | Column count | Primary purpose |
|----------------|-------------|-------------|-----------------|
| `minimal` | A | 3 | Fast baseline with minimal context |
| `temporal` | B | 14 | Calendar structure + continuous cyclical encoding + immediate lag |
| `full` | C | 86 | Maximum information content benchmark |
| `curated` | D | 15 | Balanced signal with reduced collinearity |
| `full_stable` | E | 78 | Coverage-safe high-capacity benchmark |
| `regime_profile` | F | 32 | Horizon-aware baseline-correction and regime set |

Report IV MVP execution note:
- The notebook currently evaluates these sets at `1min` resolution only.
- Multi-resolution experiments now run through the scripted Stage-6 surface rather than the notebook.
- Stage-5 also derives one non-canonical helper set for performance experiments:
  `curated_ramp`.

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
- H1 uses `minimal` as treatment against a control derived from `temporal` with
  `workday` removed (`temporal_no_workday` in notebook logic).
- H3 uses `minimal` as the simplest cross-resolution anchor in the Stage-6
  multiresolution comparison runner.

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
| `hour_sin` | Fourier | Continuous daily sine encoding |
| `hour_cos` | Fourier | Continuous daily cosine encoding |
| `dow_sin` | Fourier | Continuous weekly sine encoding |
| `dow_cos` | Fourier | Continuous weekly cosine encoding |
| `lag_1` | Lag | Previous-period load value |

Rationale:
- Adds full calendar structure while keeping feature count moderate (14 columns).
- Tests whether temporal context beyond `hour` improves predictions over the minimal set.
- Serves as the base set for H1 control derivation by removing `workday`.
- Includes continuous daily and weekly cyclical encoding so midnight and week
  boundaries remain numerically smooth for nonlinear models.

Risks:
- Calendar features (`year`, `quarter`, `month`) can be weak predictors on a 31-day
  dataset where there is minimal variation in those values.
- Some temporal features are partially redundant (e.g., `season` is determined by `month`).

### `full` (C)

Columns:
- All 89 gold columns except `timestamp`, `day_class`, and `avg_load` (target).
- Total: 86 predictor columns.

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
| `hour_sin` | Fourier | Continuous daily sine encoding |
| `hour_cos` | Fourier | Continuous daily cosine encoding |
| `dow_sin` | Fourier | Continuous weekly sine encoding |
| `dow_cos` | Fourier | Continuous weekly cosine encoding |
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
- Keeps the same daily/weekly Fourier encoding surface as `temporal` so H2
  comparisons do not accidentally conflate richer lag context with withheld
  cyclical encoding.
- Includes volatility (`rolling_std_60`) and trend direction (`slope_15`) features that
  are not available in `minimal` or `temporal`.
- Serves as the treatment condition for hypothesis H2 (lag value).

Risks:
- May omit interaction effects that matter for edge cases (e.g., seasonal load ramps
  that depend on specific rolling window sizes not included).
- `lag_1440` introduces warm-up NaN for the first 1440 periods, which reduces effective
  training data at the start of the series.

### `full_stable` (E)

Columns:
- All `full` columns except the eight longest rolling-window features:
  - `rolling_mean_240`, `rolling_std_240`, `rolling_max_240`, `rolling_min_240`
  - `rolling_mean_1440`, `rolling_std_1440`, `rolling_max_1440`, `rolling_min_1440`

Rationale:
- Preserve most of the high-capacity `full` signal while removing the features that
  cause the worst validation coverage collapse across several resolutions.
- Keeps all lag, delta, slope, and shorter rolling-window features intact, so it
  remains the closest high-capacity counterpart to `full`.

Current evidence:
- `full_stable` remains the safest large feature surface when Stage-5 wants richer
  context without reintroducing the worst long-window coverage collapse.
- In the latest targeted Stage-5 rerun, it no longer wins the top short-horizon fold
  sweep; the best promoted candidate moved back toward `curated_ramp`, which is a
  useful reminder that coverage-safe high capacity is not automatically the best holdout
  performer on every slice.

Risks:
- Still inherits the general collinearity and runtime costs of a large feature set.
- Does not fix lower-coverage issues caused by coarse-resolution `lag_1440` warm-up at
  `5min` and `10min`; it only removes the worst long rolling-window failures.

### `regime_profile` (F)

Representative columns:
- Calendar and cyclical structure: `workday`, `hour`, `season`, `time_of_day`,
  `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`
- Short memory and ramp context: `lag_1`, `lag_5`, `lag_min_15`, `lag_min_60`,
  `rolling_mean_min_60`, `rolling_std_min_60`, `slope_min_60`
- Baseline-relative context: `previous_day_load`, `avg_workday_baseline`,
  `profile_residual_lag_1`, `previous_day_residual`
- Regime flags: `prev_day_workday`, `next_day_workday`, `workday_transition`,
  `profile_activity_ratio`, `profile_active_flag`

Rationale:
- Gives the model a structured way to learn corrections to stable daily-shape baselines
  instead of relearning the entire profile from raw load alone.
- Intended for hourly and day-ahead settings where baseline-relative residuals matter
  more than very long raw lag stacks.
- Keeps the feature set compact enough for faster multiresolution and rollout iteration.

Current evidence:
- `regime_profile` is implemented end to end and measurable in Stage-5 through Stage-7.
- In the latest focused `60m` Stage-6 rerun, it did not displace `minimal`, so it is
  currently a viable challenger rather than the selected winner for that horizon.

Risks:
- If the baseline profile itself is biased, residual-derived features can propagate that
  bias unless the model is trained carefully against the right objective.
- Regime flags are only as useful as the operational day-class signal and profile
  stability in the underlying facility.

## Stage-5 Derived Helper Sets

### `curated_ramp`

Purpose:
- Extends `curated` with short-horizon ramp descriptors (`rolling_mean_3`,
  `rolling_std_3`, `ramp_flag`, `hour_x_delta_5`) for morning-ramp robustness checks.

Current evidence:
- In the latest targeted short-horizon Stage-5 run, the top promoted fold winner is
  `curated_ramp/hgb-auto-lr050-d7-leaf100-l20000-it300/residual+blend`.
- That candidate still loses the final holdout slightly to persistence, so
  `curated_ramp` is currently the strongest short-horizon challenger, not the final
  deployed winner.

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
all supported resolutions (`1s` through `15min`) produced by the pipeline. The repo
now carries both:
- legacy period-based windows, whose real-world duration scales with resolution
- time-normalized minute windows (`lag_min_*`, `rolling_*_min_*`, `slope_min_*`), whose
  lookback stays fixed across resolutions

This allows Stage-5 through Stage-7 to optimize by horizon class instead of pretending
that one shared window family is equally appropriate everywhere.

Current Report IV notebook execution still uses only `1min` resolution, while H3 is now
implemented outside the notebook through the Stage-6 multiresolution comparison runner.

