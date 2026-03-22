# Model Dataset Layer

The model dataset layer produces train, validate, and test splits from gold data. Each
split is filtered to a specific feature set and stored as an independent parquet file.
This layer enforces chronological splitting (no random shuffle) and guarantees that the
modeling target (`avg_load`) is never included as a predictor.

These datasets now feed two distinct downstream consumers:
- the canonical `1min` MVP notebook (`notebooks/003_modeling.ipynb`)
- the scripted post-MVP multiresolution stack (`scripts/005_multires_compare.py` and
  `scripts/006_recursive_rollout.py`, backed by `scripts/modeling/multires_compare.py`
  and `scripts/modeling/recursive_rollout.py`)

## Related Documents

| Document | Location |
|----------|----------|
| Architecture overview | [../000_overview/architecture.md](../000_overview/architecture.md) |
| Previous layer (Gold) | [../004_gold/gold.md](../004_gold/gold.md) |
| Feature set definitions | [feature_sets.md](../../003_modeling/feature_sets.md) |
| Hypotheses | [hypothesis.md](../../003_modeling/hypothesis.md) |
| MVMP scope | [mvmp.md](../../003_modeling/mvmp.md) |
| Pipeline operations | [pipeline.md](../../002_pipeline/pipeline.md) |
| Glossary | [glossary.md](../../004_reference/glossary.md) |

## Script

| Field | Value |
|-------|-------|
| Entry script | `scripts/003_create_model_datasets.py` |
| Canonical implementation | `scripts/stages/create_model_datasets.py` |
| Entry function | `create_model_datasets(gold_dir=None, model_dir=None, resolutions=None, feature_sets=None)` |
| Config source | `scripts/config.py` (`FEATURE_SETS`, `SPLIT_DAY_RANGES`, `TARGET_COLUMN`, `PATHS`, resolution mappings) |

## Input and Output

| Direction | Path Pattern | Format |
|-----------|-------------|--------|
| Input | `data/003_gold/power_load_{suffix}_all_features.parquet` | Apache Parquet |
| Output | `data/004_model/{suffix}_{feature_set}_{split}.parquet` | Apache Parquet |

Example output files for 5-minute resolution:

| File | Description |
|------|-------------|
| `5m_minimal_train.parquet` | Train split, minimal feature set |
| `5m_minimal_validate.parquet` | Validation split, minimal feature set |
| `5m_minimal_test.parquet` | Test split, minimal feature set |
| `5m_temporal_train.parquet` | Train split, temporal feature set |
| `5m_curated_test.parquet` | Test split, curated feature set |
| ... | One file per (resolution, feature_set, split) combination |

## Chronological Split

Time-series data must be split chronologically to prevent information leakage. The split
is defined by day order within the dataset (not by calendar date), configured in
`SPLIT_DAY_RANGES`:

| Split | Day Range | Dates (Current Dataset) | Days |
|-------|-----------|------------------------|------|
| Train | Days 1-25 | Nov 28 through Dec 22 | 25 |
| Validate | Days 26-28 | Dec 23 through Dec 25 | 3 |
| Test | Days 29-31 | Dec 26 through Dec 28 | 3 |

Known consideration: December 25 (Christmas) falls in the validation split. This is a
non-working day that may not be representative of typical validation conditions. This is
documented for awareness but not altered, as the dataset only spans 31 days.

## Validation Guarantees

The script enforces:

| Check | Description |
|-------|-------------|
| No date overlap | Train, validate, and test date sets are mutually exclusive |
| Chronological order | max(train dates) < min(validate dates) < min(test dates) |
| No target leakage | Feature sets must not include `avg_load` (raises `ValueError`) |
| Feature columns exist | All columns in the requested feature set must exist in gold |

## Feature Sets

Feature sets are defined in `scripts/config.py` (`FEATURE_SETS`) and documented in
[feature_sets.md](../../003_modeling/feature_sets.md). The current sets are:

| Name | Column Count | Purpose |
|------|-------------|---------|
| `minimal` | 3 | Fast baseline: `workday`, `hour`, `lag_1` |
| `temporal` | 14 | Calendar context + cyclical encoding + immediate lag |
| `full` | 45 | All columns except `timestamp`, `day_class`, `avg_load` |
| `curated` | 15 | Balanced selection reducing collinearity |
| `full_stable` | 37 | `full` without the unstable 240/1440 rolling windows |

See [feature_sets.md](../../003_modeling/feature_sets.md) for the complete column listing and rationale for each set.

## Output File Contents

Each output parquet file contains:

| Column Group | Columns | Description |
|-------------|---------|-------------|
| Metadata | `timestamp`, `day_class` | Always included for traceability |
| Target | `avg_load` | Modeling target, included separately |
| Features | Columns from the selected feature set | Predictor variables |

The target column `avg_load` is always present in the output but is not part of any
feature set definition. This separation is enforced by a runtime check.

## Transformation Logic

For each (resolution, feature_set) combination:

1. Read gold parquet for the resolution.
2. Parse timestamps and compute day-order mapping (day 1 = earliest date).
3. Assign each date to train, validate, or test based on `SPLIT_DAY_RANGES`.
4. Validate: no date overlap, strict chronological order.
5. Validate: feature set columns exist in gold, target column is not in feature set.
6. For each split:
   a. Filter gold rows to dates belonging to this split.
   b. Select columns: `timestamp`, `day_class`, `avg_load`, + feature set columns.
   c. Drop rows where `avg_load` is NaN (cannot evaluate without target).
   d. Sort by `timestamp`, write parquet.
   e. Log: resolution, feature set, split name, row count (before and after NaN drop),
      date range, overall null rate, per-feature null rates, target statistics
      (mean, std, min, max).

## NaN in Feature Columns

Feature columns (especially lag, rolling, and slope features) may contain NaN values
in the output files. These NaN values represent warm-up periods at the start of the
time series or at the start of each split.

The model dataset layer does not drop rows based on feature NaN -- only target NaN.
This is intentional: different modeling frameworks handle feature NaN differently
(e.g., tree-based models can use NaN natively, while linear models require imputation).
Dropping all NaN rows would significantly reduce the training set for feature sets that
include long-window features like `lag_1440` or `rolling_mean_1440`.

Per-feature null rates are logged for each split to provide visibility.

## Determinism

Model dataset outputs are deterministic: given the same gold inputs, re-running the
script produces identical parquet files. This is guaranteed by:
- Deterministic date-to-day-order mapping (sorted unique dates).
- Consistent column ordering from feature set definitions.
- Deterministic sort by `timestamp` within each split.
- No random operations.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Gold file missing for a resolution | `ValueError` with path |
| Unknown feature set name | `ValueError` naming the unknown set |
| Feature set column missing from gold | `ValueError` listing missing columns |
| Feature set includes target column | `ValueError` identifying the leakage |
| Date overlap between splits | `ValueError` identifying which splits overlap |
| Chronological order violation | `ValueError` identifying which splits are misordered |

## Hypothesis and MVMP Connection

The model datasets directly support the project hypotheses and MVMP:

| Hypothesis | Resolution | Feature Sets Compared | Metric | MVP Status |
|-----------|------------|----------------------|--------|------------|
| H1 (workday signal) | `1min` | `minimal` vs temporal-minus-workday control | MAE | Evaluated |
| H2 (lag value) | `1min` | `temporal` vs `curated` | RMSE | Evaluated |
| H3 (resolution tradeoff) | multi-resolution | `minimal` anchor in Stage-6 | MAE | Implemented outside the MVP notebook |
| H4 (nonlinear behavior, exploratory) | `1min` | all feature sets | MAE and RMSE | Evaluated |
| H5 (forecast horizon degradation) | multi-horizon | Stage-5 holdout anchor + Stage-7 challenger sweeps | MAE ratio vs persistence | Implemented outside the MVP notebook |

MVMP scope: `1min` resolution with a fixed model grid (Ridge + HGB), validation-driven
hypothesis evaluation, and one-shot holdout test reporting. See
[mvmp.md](../../003_modeling/mvmp.md) for full definition.

## Holdout Selection Policy (Current Report IV Execution)

To avoid selecting models that are only strong on small/easy subsets, holdout selection
uses evaluation coverage guardrails in the modeling notebook:

- Compute `eval_coverage = n_eval / n_eval_total` per validation row.
- Select holdout candidate from models with `eval_coverage >= 0.95`.
- If no model meets threshold, fallback to highest coverage then lowest MAE.
- Persist both `raw_best_by_mae` and `selected_for_holdout` to
  `outputs/004_modeling/<artifact_namespace>/run_manifest.json`.
- Persist `validation_mae_pct` alongside raw MAE so the recorded selection remains
  interpretable across different load scales.

This policy is especially important for feature sets with long windows
(`rolling_*_1440`), where Ridge can drop many rows due to NaN propagation.

## Post-MVP Multi-Resolution Use

The same model dataset conventions are reused by the Stage-6 through Stage-8 runners:

- Stage-6 (`scripts/005_multires_compare.py`, canonical implementation:
  `scripts/modeling/multires_compare.py`) reads gold/model-ready data across
  multiple resolutions and compares them using both native-step and matched-horizon views.
  The matched-horizon runner now evaluates both recursive one-step reuse and
  direct-endpoint horizon models, then records the winning forecast strategy alongside
  the winning resolution/feature-set/model triple. Selection is baseline-aware, so
  `avg_workday`-style baselines can win a horizon when they outperform both persistence
  and learned candidates under the configured gates.
- Stage-7 (`scripts/006_recursive_rollout.py`, canonical implementation:
  `scripts/modeling/recursive_rollout.py`) reuses a Stage-6 learned winner only when
  the requested rollout horizon exactly matches that learned winner horizon and the
  winner itself is recursive; otherwise it falls back to the configured
  resolution/feature-set/model triple to generate recursive multi-step forecast paths.
  Rollout origin selection is now explicit through `origin_policy`, with `uniform`
  as the default so day-scale evaluation uses a broader set of eligible forecast starts
  instead of midnight-only origins. It also evaluates `anchored_workday` and
  `hybrid_workday` baselines. The latest validated `1440m` evidence now promotes
  `10min/minimal/hgb-balanced` through the Stage-7 challenger sweep, while
  `avg_workday` remains the strongest baseline under the broader rollout protocol.
- Stage-8 (`scripts/008_horizon_curve.py`, canonical implementation:
  `scripts/modeling/horizon_curve.py`) consolidates the Stage-5 holdout anchor at `1m`
  and the Stage-7 challenger sweeps at longer horizons into the H5 capability envelope.
  The current curve is non-monotonic: path MAE beats persistence at `15m`, `30m`,
  `60m`, `360m`, `720m`, and `1440m`, but loses at `1m`, `120m`, and `240m`.
  That makes `120m` the clearest current weak horizon and leaves `240m` as an
  endpoint-good / path-bad mixed case.
- Stage-5 and Stage-6 both use the shared runtime defined in `config/modeling.toml`
  so parallel model-evaluation policy stays centralized instead of being duplicated
  inside individual runners.

This keeps the Report-IV notebook narrative narrow while allowing multi-resolution and
multi-step work to remain fully scripted, testable, and artifact-driven.



