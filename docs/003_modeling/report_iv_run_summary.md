# Report IV Run Summary (2026-03-04)

This document records the latest executed `1min` MVP run and its measured outcomes.
It is the factual bridge between implementation specs and report-ready interpretation.

## Run Provenance

- Pipeline rebuild: `python run_pipeline.py --stage all`
- Model datasets: `python scripts/003_create_model_datasets.py`
- Modeling notebook: `python scripts/validate_notebooks.py --notebook notebooks/003_modeling.ipynb`
- Artifact directory: `outputs/step4_artifacts/`

## Artifact Integrity

- `metrics_overall.csv`: 33 rows (`validate` + `test`)
- `metrics_by_day_class.csv`: present
- `metrics_by_hour.csv`: present
- `run_manifest.json`: present (`has_test=true`, `eval_split=validate`, `resolutions=['1min']`)
- PNG files: all four required figures present and valid PNG payloads

## Key Metrics (Validate)

- Raw MAE-best validate model: `feature_set=full`, `model=ridge`, `params=strong`
  - MAE: `435.0039`
  - RMSE: `835.1946`
  - Coverage: `1612 / 4296 = 0.3752`
- Coverage-selected holdout model: `feature_set=full`, `model=hgb`, `params=aggressive`
  - MAE: `517.4471`
  - RMSE: `949.8058`
  - Coverage: `4296 / 4296 = 1.0000`
- Validate persistence baseline:
  - MAE: `594.7170`
  - RMSE: `1195.9468`
- Best-vs-persistence improvement:
  - MAE: `+26.86%`
  - RMSE: `+30.16%`

## Hypothesis Snapshot

- H1 (workday signal): evaluated per spec via `h1_control` (`feature_set=temporal_no_workday`) vs `minimal/ridge/medium`.
  - `h1_control` MAE: `647.6705`
  - `minimal/ridge/medium` MAE: `651.7676`
  - Observed delta: `-0.63%` (target not met)
- H2 (lag/rolling value): evaluated per spec via `temporal/ridge/medium` vs `curated/ridge/medium` by RMSE.
  - `temporal/ridge/medium` RMSE: `1188.3266`
  - `curated/ridge/medium` RMSE: `1134.0887`
  - Observed delta: `+4.56%` (below `>=8%` target)
- H3 (resolution tradeoff): deferred by MVP design (single resolution `1min`)
- H4 (nonlinear behavior, exploratory): evaluated via ridge/hgb grid comparisons; no hard pass/fail threshold

## Holdout (Test) Readout

- One-shot holdout row exists for the coverage-selected model (`full/hgb/aggressive`).
- Holdout MAE: `201.3576`, RMSE: `269.0778`.
- Validate-to-test MAE shift for selected model: `-61.09%` (test period easier than validate period).
- Day-ahead extension row exists (`experiment=day_ahead`).

## Interpretation

- Implementation intent is met: deterministic run pipeline, artifacts, and holdout protocol are operational.
- Hypothesis targets are outcomes, not implementation gates:
  - H1 target not achieved in this run.
  - H2 target partially improved but below threshold.
- These findings indicate model/feature behavior to iterate on, not code-path failure.

## Spec Alignment Note

The canonical H1 control label in executed notebook artifacts is `temporal_no_workday`.
Repository docs/specs were aligned to this exact label to remove naming drift.

## Contribution Attribution

Implementation ownership for this run should be read as:

- Team (Spencer, Sean, Frank): primary project design and execution baseline (~70%)
- AI QA hardening/support: gap detection, consistency checks, and documentation alignment (~30%)

This split reflects project reality and preserves team-first credit in report framing.
