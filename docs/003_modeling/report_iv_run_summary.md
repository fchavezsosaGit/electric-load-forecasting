# Report IV Run Summary (2026-03-06)

This document records the latest executed `1min` Minimum Viable Product (MVP) run and its measured outcomes.
It is the factual bridge between implementation specs and report-ready interpretation.

## Run Provenance

- Full local E2E: `python scripts/run_e2e.py --mode full` (green)
- Full integrated rebuild: `python run_pipeline.py --stage all --include-performance --performance-mode full`
- Notebook validation: `python scripts/validate_notebooks.py`
- Full test suite: `pytest -q`
- Artifact directory: `outputs/004_modeling/`
- Performance artifact directory: `outputs/005_performance/`

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
- Coverage-selected vs persistence improvement:
  - MAE: `+12.99%` (`517.4471` vs `594.7170`)
  - RMSE: `+20.58%` (`949.8058` vs `1195.9468`)
- Raw-best vs persistence (pre-coverage-guard, 37.5% coverage -- not usable):
  - MAE: `+26.86%` (`435.0039` vs `594.7170`)
  - RMSE: `+30.16%` (`835.1946` vs `1195.9468`)

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
- Selected model holdout: MAE `201.3576`, RMSE `269.0778`.
- Persistence holdout: MAE `173.7241`, RMSE `270.9570`.
- **Persistence beats the selected model on test MAE** (173.7 vs 201.4 W); model narrowly wins on RMSE (269.1 vs 271.0 W).
- Validate-to-test MAE shift for selected model: `-61.09%` (test period easier than validate period).
- Validate-to-test MAE shift for persistence: `-70.79%` (persistence itself drops from 594.7 to 173.7 W).
- Root cause: test period (Dec 26-28) is 3 quiet post-Christmas holidays; validation has 2 working days + 1 holiday.
- Day-ahead extension row exists (`experiment=day_ahead`).

## Interpretation

- Implementation intent is met: deterministic run pipeline, artifacts, and holdout protocol are operational.
- Latest full local E2E is green end-to-end:
  - pipeline: `2611.81s`
  - notebooks: `174.87s`
  - pytest: `27.01s`
  - total: `2813.70s`
- Hypothesis targets are outcomes, not implementation gates:
  - H1 target not achieved in this run.
  - H2 target partially improved but below threshold.
- These findings indicate model/feature behavior to iterate on, not code-path failure.

## Stage-5 Performance Extension (2026-03-06)

Stage-5 artifacts (`outputs/005_performance/`) now include preflight audit,
walk-forward fold results, residual ablation, HGB coordinate search, and
causal blend guardrail diagnostics. The Stage-5 results are summarized inside
`notebooks/003_modeling.ipynb`; the repository no longer uses a separate
`004_performance.ipynb`.

Key outcomes from the latest full stage-5 run:
- Best raw fold model by ratio: `full/hgb-coordinate-lr010` with
  `fold_mean_mae_ratio=0.7723`.
- P1b acceptance model: `full/hgb-coordinate-leaf100` (meets all acceptance
  gates in `hgb_coordinate_summary.csv`).
- P2 blend for the accepted P1b model improves fold aggregate ratio from
  `0.7995` (raw) to `0.7940` (raw+blend), while max per-fold degradation remains
  below `2%`.

## Spec Alignment Note

The canonical H1 control label in executed notebook artifacts is `temporal_no_workday`.
Repository docs/specs were aligned to this exact label to remove naming drift.

## Contribution Attribution

Implementation ownership for this run should be read as:

- Team (Spencer, Sean, Frank): primary project design and execution baseline (~70%)
- AI QA hardening/support: gap detection, consistency checks, and documentation alignment (~30%)

This split reflects project reality and preserves team-first credit in report framing.
