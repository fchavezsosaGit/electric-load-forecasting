# Report IV Success Scorecard (Codebase Reality Check)

This scorecard maps the success framework in `personal/success.md` to the current
repository state and latest executed artifacts.

Date: 2026-03-04  
Scope: current codebase + latest full pipeline/modeling run  
Primary evidence: `outputs/step4_artifacts/*`, pipeline outputs in `data/*`, and
validation commands listed below.

---

## Attribution (Implementation + QA)

To accurately reflect ownership:

- **Team contribution (~70%)**: project framing, medallion architecture, hypothesis and
  MVP scope decisions, baseline implementation path, and primary modeling strategy.
- **AI support contribution (~30%)**: gap identification, consistency hardening,
  validation guardrails, and documentation synchronization.

This scorecard uses that framing and does not re-attribute core project direction.

---

## Validation Commands Executed

- `python run_pipeline.py --stage all`
- `python scripts/003_create_model_datasets.py`
- `python scripts/validate_notebooks.py`
- `python scripts/validate_notebooks.py --notebook notebooks/003_modeling.ipynb`
- `pytest`
- `pyright run_pipeline.py scripts tests`

All commands completed successfully in this review pass.

---

## Stage-by-Stage Success Status

| Stage | Success Definition (from `success.md`) | Current Evidence | Status |
|---|---|---|---|
| Project-level | Reproducible end-to-end pipeline + validated modeling path | Full pipeline + notebook re-executed; artifacts regenerated | PASS |
| Raw | Correct `.mat` structure and preservation | `P_data` shape `(86400, 31)`, `day_data` `(1, 31)`, `day_class` `(1, 31)`, NaN count `14576` | PASS |
| Bronze | Lossless long-format conversion with stable schema | `2,678,400` rows, correct timestamp span, NaN load count preserved | PASS |
| Silver | 44-column feature engineering across default resolutions | `1m/5m/10m/15m` all generated, each `44` columns | PASS |
| Gold | Null-safe modeling contract on required core columns | Gold outputs have `avg_load` null count `0` in all default resolutions | PASS |
| Model datasets | Chronological split outputs for all feature sets/resolutions | `48` parquet files (`4 resolutions x 4 feature sets x 3 splits`) | PASS |
| EDA notebooks | Executable notebooks with centralized config behavior | Full `validate_notebooks.py` completed (raw/bronze/silver profiles) | PASS |
| Modeling MVP | 1min fixed grid + baselines + holdout protocol | 24-grid + H1 + day-ahead + baselines + one-shot holdout present in artifacts | PASS |
| Hypothesis execution | H1/H2/H4 evaluated; H3 explicitly deferred | H1/H2/H4 rows present; H3 documented deferred in MVP posture | PASS |
| Evaluation integrity | Validation-driven selection; holdout not used for tuning | Coverage-guard selection policy serialized in `run_manifest.json` | PASS |
| Infra/testing | Passing tests + static typing + operational commands | `98 passed`, `pyright: 0 errors/0 warnings` | PASS |
| Documentation/governance | Current docs aligned to current behavior | Architecture/modeling/glossary/report docs refreshed in this pass | PASS |

---

## Modeling Performance Snapshot (Latest Run)

Source: `outputs/step4_artifacts/metrics_overall.csv`,
`outputs/step4_artifacts/run_manifest.json`

### Validation

- Raw MAE-best validate model: `full/ridge/strong`
  - MAE `435.0039`, RMSE `835.1946`
  - Coverage `1612 / 4296 = 0.3752` (low-coverage caveat)
- Coverage-selected holdout candidate: `full/hgb/aggressive`
  - MAE `517.4471`, RMSE `949.8058`
  - Coverage `4296 / 4296 = 1.0000`
- Validate persistence baseline: MAE `594.7170`, RMSE `1195.9468`

Improvement versus persistence:

- Raw-best vs persistence (MAE): `+26.86%`
- Coverage-selected vs persistence (MAE): `+12.99%`

### Holdout (One-shot Test)

- Selected holdout model (`full/hgb/aggressive`) test MAE: `201.3576`
- Test RMSE: `269.0778`
- Validate-to-test MAE shift (selected model): `-61.09%`

### Day-ahead Extension

- `experiment=day_ahead` row present
- MAE `2665.8329`
- Coverage `4296 / 4296 = 1.0000`

---

## Hypothesis Status Snapshot

- H1 (`h1_control` vs `minimal/ridge/medium`, MAE): `-0.63%` (target not met)
- H2 (`temporal/ridge/medium` vs `curated/ridge/medium`, RMSE): `+4.56%` (improvement, below `>=8%` target)
- H3: deferred by MVP scope (single-resolution execution)
- H4 (exploratory, curated Ridge vs HGB-conservative, MAE): `-10.09%` (Ridge better in this run)

Interpretation: implementation integrity is strong; hypothesis outcomes remain mixed,
which is a modeling finding, not a pipeline correctness failure.

---

## Remaining Risk (Not a Code Bug)

Pipeline-level analytical caveat:

- `rolling_*_1440` NaN propagation can reduce Ridge evaluation coverage for `full`
  feature set, creating non-uniform comparison subsets.

Current mitigation already implemented:

- Coverage-guard holdout selection (`MIN_VALIDATE_COVERAGE=0.95`)
- Manifest records both raw-best and selected-best for audit transparency.

---

## Definition-of-Success Readout

Against the success framework, the codebase is currently in a **deliverable-ready**
state for Report IV execution:

- Pipeline correctness and reproducibility: achieved
- Modeling/holdout protocol integrity: achieved
- Documentation and references: updated to current implementation
- Remaining work is primarily model-improvement iteration, not infrastructure repair

