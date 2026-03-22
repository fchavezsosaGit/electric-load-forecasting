# Stage Map

This document translates the repo's stable numeric stage ids into the
plain-English questions they answer.

The numeric ids are intentionally kept stable because they appear in:

- output folder names
- run manifests
- report citations
- older experiment references

The practical fix is not to rename them; it is to pair them with a readable map.

## High-Level Flow

1. Build trusted data layers.
2. Establish the canonical `1min` benchmark surface.
3. Promote only the short-horizon candidates that survive holdout.
4. Compare matched horizons across resolutions.
5. Select recursive rollout policies by horizon objective.
6. Summarize the capability envelope across horizons.
7. Replay the stacked control policy end to end.

## Stage Directory Map

| Stable stage id | Plain-English name | Main question | Primary outputs |
|---|---|---|---|
| Stage-0 | Raw ingest contract | Is the source payload structurally valid and preserved? | `data/000_raw/` |
| Stage-1 | Bronze conversion | Did raw MATLAB data become a timestamped long-format parquet cleanly? | `data/001_bronze/` |
| Stage-2 | Silver feature layer | Did we resample correctly and build the feature surface we expect? | `data/002_silver/` |
| Stage-3 | Gold and model datasets | Do we have null-safe, leakage-safe train/validate/test datasets? | `data/003_gold/`, `data/004_model/` |
| Stage-4 | Notebook benchmark surface | What does the canonical `1min` benchmark look like in notebook form? | `outputs/004_modeling/` |
| Stage-5 | Short-horizon holdout gate | Does any learned `1min` challenger actually beat persistence on holdout? | `outputs/005_performance/` |
| Stage-6 | Matched-horizon comparison | At the same representable horizon, which resolution/model pair wins? | `outputs/006_multires/` |
| Stage-7 | Rollout selection and sweeps | Which recursive rollout policy wins for the requested horizon objective? | `outputs/007_rollout/` |
| Stage-8 | Horizon capability curve | Across horizons, where do learned policies help and where do baselines still win? | `outputs/009_horizon_curve/` |
| Stage-10 | Forecast-control backtest | Does the full day-ahead plus intraday update stack improve real control outcomes? | `outputs/010_forecast_control/` |

## Support Surfaces

Not every numbered output folder is a separate modeling decision stage.

- `outputs/008_notebook_runs/` is an evidence archive surface, not a separate
  forecast-selection stage. It preserves executed notebook snapshots and figure
  metadata before tracked notebook outputs are cleared.
- There is no separate Stage-9 decision layer in the current repo. The
  numbering stays aligned with the established artifact/report contract rather
  than being renumbered after later pipeline additions.

## What To Read First

If you want the shortest path to "what are we doing now?":

1. [current_validation_snapshot.md](../003_modeling/current_validation_snapshot.md)
2. [current_operating_approach.md](../003_modeling/current_operating_approach.md)
3. [model_and_blend_guide.md](../003_modeling/model_and_blend_guide.md)
4. [`outputs/010_forecast_control/commercial_facility/latest/control_policy.json`](../../outputs/010_forecast_control/commercial_facility/latest/control_policy.json)
5. [`outputs/010_forecast_control/commercial_facility/latest/current_evidence_index.md`](../../outputs/010_forecast_control/commercial_facility/latest/current_evidence_index.md)
6. [`outputs/009_horizon_curve/commercial_facility/latest/horizon_curve_summary.md`](../../outputs/009_horizon_curve/commercial_facility/latest/horizon_curve_summary.md)

If you want the shortest path to "how does the pipeline work?":

1. [pipeline.md](pipeline.md)
2. [architecture.md](../001_architecture/000_overview/architecture.md)
3. [README.md](../../README.md)

## Current Operating Interpretation

The repo is no longer a single-model leaderboard.

The current evidence supports a layered operating stack:

1. freeze a `24h` day-ahead anchor
2. refresh that path intraday when the evidence says the refresh helps
3. apply an hourly corrective layer
4. apply a `15m` phase layer only if it still helps after the hourly layer
5. apply the final `1m` nowcast separately

That is why Stage-5, Stage-8, and Stage-10 all exist. They answer different
questions:

- Stage-5 asks whether a learned short-horizon anchor is operationally credible
- Stage-8 asks which horizon winners exist by objective
- Stage-10 asks whether the whole stack improves the actual control outcome
