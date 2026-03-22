"""Current-validation snapshot generation tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def _find_project_root(start: Path) -> Path:
    """Find the repository root so the snapshot helper can be imported in tests."""
    for candidate in [start, *start.parents]:
        if (candidate / "run_pipeline.py").exists() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Unable to locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_snapshot_module():
    """Load the snapshot wrapper module directly from disk for isolated tests."""
    path = SCRIPTS_DIR / "write_validation_snapshot.py"
    spec = importlib.util.spec_from_file_location("test_write_validation_snapshot_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    """Write one JSON fixture file for the synthetic artifact surface."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write one CSV fixture file for the synthetic artifact surface."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_write_validation_snapshot_generates_canonical_markdown(tmp_path):
    """Build the current-state markdown from the latest artifact surfaces."""
    module = _load_snapshot_module()
    artifact_namespace = "commercial_facility"

    latest_roots = {
        "performance": tmp_path / "outputs" / "005_performance" / artifact_namespace / "latest",
        "multires": tmp_path / "outputs" / "006_multires" / artifact_namespace / "latest",
        "rollout": tmp_path / "outputs" / "007_rollout" / artifact_namespace / "latest",
        "rollout_sweep": tmp_path / "outputs" / "007_rollout" / artifact_namespace / "challenger_sweeps" / "latest",
        "notebooks": tmp_path / "outputs" / "008_notebook_runs" / artifact_namespace / "latest",
        "horizon_curve": tmp_path / "outputs" / "009_horizon_curve" / artifact_namespace / "latest",
        "forecast_control": tmp_path / "outputs" / "010_forecast_control" / artifact_namespace / "latest",
    }
    for root in latest_roots.values():
        root.mkdir(parents=True, exist_ok=True)

    for stage_name, root in latest_roots.items():
        _write_json(root / "run_manifest.json", {"status": "success", "run_id": f"{stage_name}_run"})

    _write_json(
        latest_roots["performance"] / "deployment_recommendation.json",
        {
            "recommended_candidate_label": "persistence",
            "decision_reason": "learned 1m superiority is not supported by the current holdout evidence.",
        },
    )
    _write_csv(
        latest_roots["performance"] / "holdout_evaluation.csv",
        [
            {
                "candidate_label": "persistence",
                "candidate_type": "baseline",
                "mae": 173.724099,
                "mae_pct": 8.380502,
            },
            {
                "candidate_label": "curated_ramp/hgb-frontier-lr010-leaf100/residual+blend",
                "candidate_type": "learned",
                "mae": 174.891820,
                "mae_pct": 8.436833,
            },
        ],
    )
    _write_json(
        latest_roots["rollout_sweep"] / "recommended_candidate.json",
        {
            "recommended_resolution": "10min",
            "recommended_feature_set": "minimal",
            "recommended_model_label": "hgb-balanced",
            "recommended_target_mode": "raw",
            "selection_target": "profile_shape_mae",
            "recommended_metric_value": 717.777613,
            "recommended_metric_pct": 36.245099,
        },
    )
    _write_csv(
        latest_roots["multires"] / "selection_summary.csv",
        [
            {
                "use_case": "matched_horizon_15m",
                "winner_type": "baseline_model",
                "winner_resolution": "1min",
                "winner_feature_set": "baseline",
                "winner_model_label": "persistence",
                "winner_forecast_strategy": "path_baseline",
                "winner_horizon_minutes": 15,
            },
            {
                "use_case": "matched_horizon_60m",
                "winner_type": "learned_model",
                "winner_resolution": "1min",
                "winner_feature_set": "minimal",
                "winner_model_label": "hgb-balanced",
                "winner_forecast_strategy": "recursive",
                "winner_horizon_minutes": 60,
            },
        ],
    )
    _write_csv(
        latest_roots["multires"] / "matched_horizon_metrics.csv",
        [
            {
                "comparison_mode": "matched_horizon",
                "resolution": "30s",
                "horizon_minutes": 15,
                "feature_set": "curated",
                "model_label": "xgb-balanced",
                "baseline_label": "persistence",
                "candidate_type": "learned",
                "forecast_strategy": "recursive",
                "mae": 1251.906786,
                "rmse": 1564.382745,
                "mae_pct": 37.103006,
                "path_mae": 1392.568323,
                "path_mae_pct": 39.308929,
                "mae_ratio_to_persistence": 0.827315,
                "rmse_ratio_to_persistence": 0.906029,
                "fold_std_mae_ratio": 0.287154,
                "n_eval": 240,
                "eval_coverage": 1.0,
                "runtime_seconds": 1.804775,
                "fold_n": 2,
                "source_mode": "bronze_direct",
                "eligible": False,
                "practical_gain_passed": True,
                "pareto_passed": False,
            },
            {
                "comparison_mode": "matched_horizon",
                "resolution": "1min",
                "horizon_minutes": 15,
                "feature_set": "curated",
                "model_label": "hgb-balanced",
                "baseline_label": "persistence",
                "candidate_type": "learned",
                "forecast_strategy": "recursive",
                "mae": 1391.319716,
                "rmse": 1675.461472,
                "mae_pct": 41.883327,
                "path_mae": 1054.291196,
                "path_mae_pct": 30.339071,
                "mae_ratio_to_persistence": 0.959857,
                "rmse_ratio_to_persistence": 1.013784,
                "fold_std_mae_ratio": 0.148024,
                "n_eval": 120,
                "eval_coverage": 1.0,
                "runtime_seconds": 0.472030,
                "fold_n": 2,
                "source_mode": "bronze_direct",
                "eligible": True,
                "practical_gain_passed": True,
                "pareto_passed": False,
            },
        ],
    )
    _write_csv(
        latest_roots["horizon_curve"] / "horizon_curve_summary.csv",
        [
            {
                "horizon_minutes": 1,
                "selection_target": "endpoint_mae",
                "candidate_label": "curated_ramp/hgb-frontier-lr010-leaf100/residual+blend",
                "learned_endpoint_mae": 174.891820,
                "persistence_endpoint_mae": 173.724099,
            },
            {
                "horizon_minutes": 15,
                "selection_target": "next_lock_mae",
                "candidate_label": "hgb-balanced::phase_bucket_next_lock_policy",
                "learned_next_lock_mae": 266.837858,
                "persistence_next_lock_mae": 434.846944,
            },
            {
                "horizon_minutes": 60,
                "selection_target": "next_lock_mae",
                "candidate_label": "cross_candidate_portfolio::phase_bucket_next_lock_policy",
                "learned_next_lock_mae": 253.104260,
                "persistence_next_lock_mae": 379.116458,
            },
            {
                "horizon_minutes": 1440,
                "selection_target": "profile_shape_mae",
                "candidate_label": "hgb-balanced::raw",
                "learned_profile_shape_mae": 717.777613,
                "persistence_profile_shape_mae": 746.527115,
            },
        ],
    )
    _write_json(
        latest_roots["forecast_control"] / "control_policy.json",
        {
            "day_ahead": {"candidate_label": "5min/minimal/hgb-frontier-lr010-l2001::hybrid_workday_residual"},
            "hourly": {"candidate_label": "10min/minimal/hgb-balanced::hybrid_phase_gate"},
            "phase": {"stack_guard_applied_candidate_label": "5min/full/persistence"},
            "nowcast_anchor": {"candidate_label": "persistence"},
            "day_ahead_refresh": {
                "recommended_policy": "unconditional_refresh",
                "trigger_mode": "residual_or_activity",
                "evaluation_trigger_rate": 0.391304,
                "rolling_benchmark": {"trigger_rate": 0.421569},
            },
        },
    )
    _write_json(
        latest_roots["forecast_control"] / "optimizer_delivery_contract.json",
        {
            "contract_version": "1.1",
            "cadence_minutes": 15,
            "selected_layer_priority": ["nowcast", "phase", "hourly", "day_ahead"],
            "freshness": {
                "row_fields": [
                    "as_of_timestamp",
                    "effective_forecast_as_of",
                    "forecast_age_minutes",
                    "stale_threshold_minutes",
                    "is_stale_forecast",
                ]
            },
            "uncertainty": {"method": "empirical_residual_quantiles"},
            "confidence_signal": {"type": "heuristic_operational_trust_score"},
        },
    )
    _write_json(
        latest_roots["forecast_control"] / "optimizer_operational_policy.json",
        {
            "hardware_policy": {
                "portable_default": (
                    "CPU-safe HGB and baseline paths remain the default-safe contract for non-accelerated and ARM64 hosts."
                )
            }
        },
    )
    _write_csv(
        latest_roots["forecast_control"] / "control_backtest_summary.csv",
        [
            {"layer": "day_ahead_frozen", "lock_mae": 792.181615, "profile_shape_mae": 792.884074},
            {"layer": "after_hourly_updates", "lock_mae": 590.696303, "profile_shape_mae": 711.905007},
            {"layer": "after_phase_updates", "lock_mae": 473.529740, "profile_shape_mae": 629.058953},
            {"layer": "after_nowcast_updates", "lock_mae": 48.542732, "profile_shape_mae": 175.332271},
        ],
    )
    _write_csv(
        latest_roots["forecast_control"] / "rolling_control_backtest_summary.csv",
        [
            {"layer": "day_ahead_frozen", "lock_mae": 776.285484, "profile_shape_mae": 791.553910},
            {"layer": "after_hourly_updates", "lock_mae": 611.746724, "profile_shape_mae": 734.537622},
            {"layer": "after_phase_updates", "lock_mae": 611.746724, "profile_shape_mae": 734.537622},
            {"layer": "after_nowcast_updates", "lock_mae": 47.818068, "profile_shape_mae": 175.050819},
        ],
    )
    _write_csv(
        latest_roots["forecast_control"] / "day_ahead_refresh_summary.csv",
        [
            {"scenario": "frozen_day_ahead", "profile_shape_mae": 792.884074},
            {"scenario": "unconditional_refresh", "profile_shape_mae": 755.593168},
            {"scenario": "triggered_refresh", "profile_shape_mae": 771.146369},
        ],
    )
    _write_csv(
        latest_roots["forecast_control"] / "optimizer_delivery_uncertainty_summary.csv",
        [
            {"scope": "all_intervals", "interval_80_coverage": 0.8125, "interval_95_coverage": 0.9375}
        ],
    )
    _write_csv(
        latest_roots["forecast_control"] / "optimizer_delivery_preview.csv",
        [
            {
                "cycle_origin_timestamp": "2026-03-01T00:00:00",
                "interval_start": "2026-03-01T00:00:00",
                "interval_end": "2026-03-01T00:15:00",
                "forecast_value": 100.0,
            }
        ],
    )
    _write_json(
        latest_roots["notebooks"] / "run_manifest.json",
        {
            "status": "success",
            "notebooks": [
                {
                    "source_path": "notebooks/003_modeling.ipynb",
                    "output_count": 42,
                    "artifact_validation": {
                        "csv_artifacts": {
                            "metrics_overall.csv": {"rows": 57},
                            "metrics_by_day_class.csv": {"rows": 98},
                            "metrics_by_hour.csv": {"rows": 1176},
                        }
                    },
                }
            ],
        },
    )
    (latest_roots["forecast_control"] / "current_evidence_index.md").write_text(
        "# Current Evidence Index\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "docs" / "003_modeling" / "current_validation_snapshot.md"
    written = module.write_validation_snapshot(
        project_root=tmp_path,
        artifact_namespace=artifact_namespace,
        output_path=output_path,
        step_seconds={"pipeline": 7017.78, "notebooks": 548.56, "pytest": 281.63},
        generated_at_utc="2026-03-14T12:34:56+00:00",
    )

    content = written.read_text(encoding="utf-8")
    assert written == output_path.resolve()
    assert "# Current Validation Snapshot" in content
    assert "The repo still does not support a learned-superiority claim at `1m`." in content
    assert "does not yet show a robust rolling gain beyond the hourly layer" in content
    assert "`10min/minimal/hgb-balanced::raw` on `profile_shape_mae` with 717.777613 (36.245099%)" in content
    assert "## How To Read The Winners" in content
    assert "Its `1m` row does not override the Stage-5 deployment recommendation by itself." in content
    assert "[Model and Blend Guide](model_and_blend_guide.md)" in content
    assert "## Current Resolution Policy" in content
    assert "- Current optimizer-facing actual resolution: `1min` with `15` minute lock intervals." in content
    assert "Best current sub-minute challenger: `30s/curated/xgb-balanced/recursive` at `15m` with MAE ratio `0.827315` to persistence." in content
    assert "the best current sub-minute candidate still fails the Stage-6 operating gates" in content
    assert "`fold_std_mae_ratio=0.287154` against the configured stability gate `0.200000`" in content
    assert "- Day-ahead: `5min/minimal/hgb-frontier-lr010-l2001::hybrid_workday_residual`" in content
    assert "## Optimizer Delivery Surface" in content
    assert "Operational policy:" in content
    assert "Contract version: `1.1`" in content
    assert "Uncertainty method: `empirical_residual_quantiles`" in content
    assert "Confidence signal: `heuristic_operational_trust_score`" in content
    assert "Preview rows carry freshness fields:" in content
    assert "Runtime portability: `CPU-safe HGB and baseline paths remain the default-safe contract for non-accelerated and ARM64 hosts.`" in content
    assert "- Recommended policy: `unconditional_refresh`" in content
    assert "Pipeline: `7017.78s`" in content
    assert "../../outputs/010_forecast_control/commercial_facility/latest/current_evidence_index.md" in content
    assert "[Current Operating Approach](current_operating_approach.md)" in content
    assert "## High-Signal Visual Anchors" in content
    assert "../../outputs/009_horizon_curve/commercial_facility/latest/fig_horizon_ratio_curve.png" in content
    assert "[Current Visualization Guide](current_visualization_guide.md)" in content
    assert "../../outputs/reports/commercial_facility/latest/validation_dashboard.html" in content


def test_phase_interpretation_line_prefers_positive_rolling_inference():
    """Report a positive phase message only when rolling inference supports it."""
    module = _load_snapshot_module()

    rolling_summary = pd.DataFrame(
        [
            {"layer": "after_hourly_updates", "lock_mae": 589.334141},
            {"layer": "after_phase_updates", "lock_mae": 479.598360},
        ]
    )
    rolling_inference = pd.DataFrame(
        [
            {
                "scope": "rolling_evaluation",
                "comparison_label": "phase_vs_hourly",
                "metric_name": "lock_mae",
                "gain_metric": 109.735781,
                "gain_ci_excludes_zero": True,
                "candidate_better_than_baseline": True,
            }
        ]
    )

    line = module._phase_interpretation_line(
        rolling_summary=rolling_summary,
        rolling_inference=rolling_inference,
    )

    assert "adds a meaningful rolling gain" in line


def test_phase_interpretation_line_reports_flat_phase_when_inference_is_not_supportive():
    """Keep the narrative conservative when the rolling phase comparison is flat."""
    module = _load_snapshot_module()

    rolling_summary = pd.DataFrame(
        [
            {"layer": "after_hourly_updates", "lock_mae": 611.746724},
            {"layer": "after_phase_updates", "lock_mae": 611.746724},
        ]
    )
    rolling_inference = pd.DataFrame(
        [
            {
                "scope": "rolling_evaluation",
                "comparison_label": "phase_vs_hourly",
                "metric_name": "lock_mae",
                "gain_metric": 0.0,
                "gain_ci_excludes_zero": False,
                "candidate_better_than_baseline": False,
            }
        ]
    )

    line = module._phase_interpretation_line(
        rolling_summary=rolling_summary,
        rolling_inference=rolling_inference,
    )

    assert "does not yet show a robust rolling gain beyond the hourly layer" in line
