================================================================================
Plan: Data Pipeline And Model Readiness
================================================================================

> Historical note:
> This document is retained as the foundation build plan for the medallion and
> model-dataset pipeline. It is not the active optimizer-facing modeling
> roadmap. For the current direction, use
> [002_operating_direction_spec.md](../000_governance/002_operating_direction_spec.md),
> [operational_hypotheses.md](../003_modeling/operational_hypotheses.md), and
> [current_operating_approach.md](../003_modeling/current_operating_approach.md).

Canonical specifications:
- [000_spec.md](../000_governance/000_spec.md)
- [001_spec.md](../000_governance/001_spec.md)
- [002_operating_direction_spec.md](../000_governance/002_operating_direction_spec.md)

Glossary:
- [glossary.md](../004_reference/glossary.md)

================================================================================
Step 1: Understand raw structure
================================================================================
- Input file: data/000_raw/P_data.mat
- Arrays: P_data, day_data, day_class
- Expected date range in current dataset: 2025-11-28 to 2025-12-28

================================================================================
Step 2: Raw -> Bronze
================================================================================
Input:
- data/000_raw/P_data.mat

Output:
- data/001_bronze/power_load_1s.parquet

Schema:
- timestamp
- day_class
- load

================================================================================
Step 3: Bronze -> Silver
================================================================================
Input:
- data/001_bronze/power_load_1s.parquet

Supported resolutions:
- 1s
- 5s
- 10s
- 30s
- 60s (alias of 1min)
- 1min
- 5min
- 10min
- 15min

Default resolutions:
- 1min
- 5min
- 10min
- 15min

Billing safety note:
- If interval billing settles on 15-minute windows, generate 15-minute outputs from
  the beginning of processing to avoid expensive post-hoc reconciliation risk.

Silver outputs use suffixes:
- power_load_1s.parquet
- power_load_5s.parquet
- power_load_10s.parquet
- power_load_30s.parquet
- power_load_1m.parquet
- power_load_5m.parquet
- power_load_10m.parquet
- power_load_15m.parquet

Silver schema:
- 82 columns (core + temporal + Fourier + workday + lag + rolling + delta + slope + time-normalized windows + baseline/regime context)

================================================================================
Step 4: Silver -> Gold
================================================================================
Gold outputs mirror silver resolutions:
- data/003_gold/power_load_{suffix}_all_features.parquet

Gold behavior:
- preserve schema
- drop rows with nulls in required core modeling columns
- keep deterministic ordering

================================================================================
Step 5: Gold -> Model datasets
================================================================================
Script:
- scripts/003_create_model_datasets.py

Feature set source:
- [feature_sets.md](../003_modeling/feature_sets.md)

Output naming:
- data/004_model/{suffix}_{feature_set}_{split}.parquet

Target column:
- avg_load (target only; not part of predictor feature sets)

================================================================================
Step 6: Train / Validate / Test split
================================================================================
Chronological split by day order:
- train: 1-25
- validate: 26-28
- test: 29-31

Validation purpose:
- model selection and tuning without touching final test data

================================================================================
Step 7: Evaluation
================================================================================
Primary metric:
- MAE

Secondary metrics:
- RMSE
- Peak-load error
- Daily energy error

================================================================================
Step 8: Documentation and governance
================================================================================
- README is onboarding/runbook.
- [000_spec.md](../000_governance/000_spec.md) is source-of-truth spec.
- [changelog.md](../../changelog.md) serves as index to spec-specific changelogs.
- [glossary.md](../004_reference/glossary.md) standardizes vocabulary across all docs.

