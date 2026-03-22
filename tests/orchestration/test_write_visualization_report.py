"""Integrated visualization-report generation tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def _find_project_root(start: Path) -> Path:
    """Find the repository root so the visualization helper can be imported in tests."""
    for candidate in [start, *start.parents]:
        if (candidate / "run_pipeline.py").exists() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Unable to locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_visualization_module():
    """Load the visualization wrapper module directly from disk for isolated tests."""
    path = SCRIPTS_DIR / "write_visualization_report.py"
    spec = importlib.util.spec_from_file_location("test_write_visualization_report_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    """Write one JSON fixture file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write one CSV fixture file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_png(path: Path) -> None:
    """Write a tiny valid PNG fixture file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\nIDATx\x9cc\xf8\x0f\x00\x01\x01\x01\x00\x18\xdd\x8d\xb1"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def test_write_visualization_report_generates_dashboard_and_guide(tmp_path):
    """Build the integrated guide and HTML dashboard from the latest artifacts."""
    module = _load_visualization_module()
    artifact_namespace = "commercial_facility"

    roots = {
        "modeling": tmp_path / "outputs" / "004_modeling" / artifact_namespace,
        "performance": tmp_path / "outputs" / "005_performance" / artifact_namespace / "latest",
        "multires": tmp_path / "outputs" / "006_multires" / artifact_namespace / "latest",
        "rollout": tmp_path / "outputs" / "007_rollout" / artifact_namespace / "latest",
        "rollout_sweep": tmp_path / "outputs" / "007_rollout" / artifact_namespace / "challenger_sweeps" / "latest",
        "notebooks": tmp_path / "outputs" / "008_notebook_runs" / artifact_namespace / "latest",
        "horizon_curve": tmp_path / "outputs" / "009_horizon_curve" / artifact_namespace / "latest",
        "forecast_control": tmp_path / "outputs" / "010_forecast_control" / artifact_namespace / "latest",
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)

    for stage_name, root in roots.items():
        _write_json(root / "run_manifest.json", {"status": "success", "run_id": f"{stage_name}_run"})

    _write_png(roots["modeling"] / "fig_actual_vs_predicted.png")
    _write_png(roots["modeling"] / "fig_model_comparison.png")
    (roots["modeling"] / "figure_guide.md").write_text(
        "\n".join(
            [
                "# Visualization Guide",
                "",
                "## Stage-4 Notebook Figures",
                "",
                "These figures explain the benchmark surface before promotion.",
                "",
                "### `fig_actual_vs_predicted.png`: Actual vs predicted",
                "",
                "- Intent: Show overall fit quality on the benchmark surface.",
                "- How to read it: The closer the prediction track sits to the actual line, the better the fit.",
                "- What to look for: Large sustained departures, especially around regime changes.",
                "",
                "### `fig_model_comparison.png`: Model comparison",
                "",
                "- Intent: Show benchmark ranking across the main Stage-4 candidates.",
                "- How to read it: Compare the bars across learned and baseline rows.",
                "- What to look for: Whether learned models beat persistence without leaning on low-coverage rows.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    _write_json(
        roots["performance"] / "deployment_recommendation.json",
        {
            "recommended_candidate_label": "persistence",
            "decision_reason": "learned 1m superiority is not yet supported on holdout.",
        },
    )
    _write_csv(
        roots["performance"] / "holdout_evaluation.csv",
        [
            {
                "candidate_label": "persistence",
                "candidate_type": "baseline",
                "mae": 173.724099,
                "mae_pct": 8.380502,
                "mae_ratio_to_persistence": 1.0,
            },
            {
                "candidate_label": "curated_ramp/hgb-balanced/residual+blend",
                "candidate_type": "promoted_learned",
                "mae": 175.228231,
                "mae_pct": 8.453062,
                "mae_ratio_to_persistence": 1.008658,
            },
            {
                "candidate_label": "anchored_workday",
                "candidate_type": "baseline",
                "mae": 257.954664,
                "mae_pct": 12.443809,
                "mae_ratio_to_persistence": 1.484853,
            },
        ],
    )

    _write_csv(
        roots["multires"] / "matched_horizon_metrics.csv",
        [
            {
                "comparison_mode": "matched_horizon",
                "resolution": "1min",
                "horizon_minutes": 15,
                "feature_set": "curated",
                "model_label": "hgb-balanced",
                "candidate_type": "learned",
                "forecast_strategy": "recursive",
                "mae_ratio_to_persistence": 0.959857,
                "eval_coverage": 1.0,
                "runtime_seconds": 0.432909,
                "eligible": True,
            },
            {
                "comparison_mode": "matched_horizon",
                "resolution": "30s",
                "horizon_minutes": 15,
                "feature_set": "curated",
                "model_label": "xgb-balanced",
                "candidate_type": "learned",
                "forecast_strategy": "recursive",
                "mae_ratio_to_persistence": 0.827315,
                "eval_coverage": 1.0,
                "runtime_seconds": 2.526453,
                "eligible": False,
            },
        ],
    )

    _write_csv(
        roots["rollout"] / "rollout_selection_summary.csv",
        [
            {
                "selection_target": "path_mae",
                "winner_candidate_label": "hgb-balanced::hybrid_workday_residual",
                "winner_metric_value": 782.772447,
                "winner_metric_pct": 39.527096,
                "origin_n": 8,
                "decision_reason": "Lowest path MAE across rollout candidates.",
            },
            {
                "selection_target": "next_lock_mae",
                "winner_candidate_label": "hgb-balanced::raw",
                "winner_metric_value": 409.047215,
                "winner_metric_pct": 19.456219,
                "origin_n": 8,
                "decision_reason": "Lowest next 15-minute MAE across rollout candidates.",
            },
        ],
    )

    _write_csv(
        roots["horizon_curve"] / "horizon_curve_summary.csv",
        [
            {
                "horizon_minutes": 1,
                "selection_target": "endpoint_mae",
                "candidate_label": "curated_ramp/hgb-balanced/residual+blend",
                "learned_endpoint_mae": 175.228231,
                "persistence_endpoint_mae": 173.724099,
                "best_baseline_endpoint_mae": 173.724099,
                "beats_persistence_endpoint": False,
                "beats_persistence_path": False,
                "beats_persistence_phase": False,
                "beats_persistence_next_lock": False,
                "beats_persistence_profile_shape": False,
            },
            {
                "horizon_minutes": 15,
                "selection_target": "next_lock_mae",
                "candidate_label": "hgb-balanced::phase_bucket_next_lock_policy",
                "learned_next_lock_mae": 266.837858,
                "persistence_next_lock_mae": 434.846944,
                "best_baseline_next_lock_mae": 419.789637,
                "beats_persistence_endpoint": False,
                "beats_persistence_path": True,
                "beats_persistence_phase": True,
                "beats_persistence_next_lock": True,
                "beats_persistence_profile_shape": True,
            },
            {
                "horizon_minutes": 1440,
                "selection_target": "profile_shape_mae",
                "candidate_label": "hgb-balanced::raw",
                "learned_profile_shape_mae": 717.777613,
                "persistence_profile_shape_mae": 746.527115,
                "best_baseline_profile_shape_mae": 746.527115,
                "beats_persistence_endpoint": False,
                "beats_persistence_path": False,
                "beats_persistence_phase": False,
                "beats_persistence_next_lock": False,
                "beats_persistence_profile_shape": True,
            },
        ],
    )

    _write_csv(
        roots["forecast_control"] / "control_backtest_summary.csv",
        [
            {"layer": "day_ahead_frozen", "lock_mae": 767.411283, "profile_shape_mae": 788.533702},
            {"layer": "after_hourly_updates", "lock_mae": 592.584681, "profile_shape_mae": 719.706066},
            {"layer": "after_phase_updates", "lock_mae": 561.308413, "profile_shape_mae": 688.618038},
            {"layer": "after_nowcast_updates", "lock_mae": 47.315636, "profile_shape_mae": 175.058501},
        ],
    )
    _write_csv(
        roots["forecast_control"] / "rolling_control_backtest_summary.csv",
        [
            {"layer": "day_ahead_frozen", "lock_mae": 763.962699, "profile_shape_mae": 786.255244},
            {"layer": "after_hourly_updates", "lock_mae": 589.334141, "profile_shape_mae": 718.573048},
            {"layer": "after_phase_updates", "lock_mae": 479.598360, "profile_shape_mae": 609.081322},
            {"layer": "after_nowcast_updates", "lock_mae": 47.285804, "profile_shape_mae": 175.301250},
        ],
    )
    _write_csv(
        roots["forecast_control"] / "control_backtest_by_cycle.csv",
        [
            {
                "day_ahead_lock_mae": 814.072697,
                "day_ahead_profile_shape_mae": 805.271296,
                "hourly_lock_mae": 676.086732,
                "hourly_profile_shape_mae": 769.196835,
                "phase_lock_mae": 635.791285,
                "phase_profile_shape_mae": 725.753868,
                "nowcast_lock_mae": 50.090076,
                "nowcast_profile_shape_mae": 171.198466,
            },
            {
                "day_ahead_lock_mae": 651.192936,
                "day_ahead_profile_shape_mae": 704.544500,
                "hourly_lock_mae": 515.136919,
                "hourly_profile_shape_mae": 642.415894,
                "phase_lock_mae": 492.225271,
                "phase_profile_shape_mae": 625.685007,
                "nowcast_lock_mae": 45.181984,
                "nowcast_profile_shape_mae": 167.328340,
            },
        ],
    )
    _write_csv(
        roots["forecast_control"] / "rolling_control_backtest_by_cycle.csv",
        [
            {
                "day_ahead_lock_mae": 814.072697,
                "day_ahead_profile_shape_mae": 805.271296,
                "hourly_lock_mae": 676.086732,
                "hourly_profile_shape_mae": 769.196835,
                "phase_lock_mae": 494.319500,
                "phase_profile_shape_mae": 615.352621,
                "nowcast_lock_mae": 50.090076,
                "nowcast_profile_shape_mae": 171.198466,
            },
            {
                "day_ahead_lock_mae": 651.192936,
                "day_ahead_profile_shape_mae": 704.544500,
                "hourly_lock_mae": 515.136919,
                "hourly_profile_shape_mae": 642.415894,
                "phase_lock_mae": 383.063373,
                "phase_profile_shape_mae": 539.266180,
                "nowcast_lock_mae": 45.181984,
                "nowcast_profile_shape_mae": 167.328340,
            },
        ],
    )
    _write_csv(
        roots["forecast_control"] / "day_ahead_refresh_summary.csv",
        [
            {
                "scenario": "frozen_day_ahead",
                "lock_mae": 767.411283,
                "profile_shape_mae": 788.533702,
                "refresh_update_count": 0,
                "lock_mae_gain_vs_frozen": 0.0,
                "profile_shape_mae_gain_vs_frozen": 0.0,
            },
            {
                "scenario": "unconditional_refresh",
                "lock_mae": 606.603723,
                "profile_shape_mae": 701.862380,
                "refresh_update_count": 23,
                "lock_mae_gain_vs_frozen": 160.807560,
                "profile_shape_mae_gain_vs_frozen": 86.671323,
            },
            {
                "scenario": "triggered_refresh",
                "lock_mae": 655.385169,
                "profile_shape_mae": 732.516445,
                "refresh_update_count": 9,
                "lock_mae_gain_vs_frozen": 112.026114,
                "profile_shape_mae_gain_vs_frozen": 56.017257,
            },
        ],
    )
    _write_csv(
        roots["forecast_control"] / "rolling_control_layer_inference.csv",
        [
            {
                "scope": "rolling_evaluation",
                "comparison_label": "phase_vs_hourly",
                "metric_name": "lock_mae",
                "candidate_better_than_baseline": True,
                "gain_ci_excludes_zero": True,
            }
        ],
    )

    guide_path = tmp_path / "docs" / "003_modeling" / "current_visualization_guide.md"
    dashboard_path = tmp_path / "outputs" / "reports" / artifact_namespace / "latest" / "validation_dashboard.html"
    result = module.write_visualization_report(
        project_root=tmp_path,
        artifact_namespace=artifact_namespace,
        doc_output_path=guide_path,
        dashboard_output_path=dashboard_path,
        generated_at_utc="2026-03-19T00:00:00+00:00",
        embed_plotlyjs=False,
    )

    guide_content = result.guide_path.read_text(encoding="utf-8")
    dashboard_content = result.dashboard_path.read_text(encoding="utf-8")

    assert result.guide_path == guide_path.resolve()
    assert result.dashboard_path == dashboard_path.resolve()
    assert "# Current Visualization Guide" in guide_content
    assert "## Primary Markdown Embeds" in guide_content
    assert "Embed tier" in guide_content
    assert "Decision question" in guide_content
    assert "Core inline" in guide_content
    assert "## Integrated Visuals" in guide_content
    assert "1-minute Holdout Leaderboard" in guide_content
    assert "Stage-7 Objective Winners" in guide_content
    assert "fig_actual_vs_predicted.png" in guide_content
    assert "fig_model_comparison.png" in guide_content
    assert "Electric Load Forecasting Visual Story" in dashboard_content
    assert "What Success Means Right Now" in dashboard_content
    assert "Matched-Horizon Runtime vs Persistence" in dashboard_content
    assert "Which Visuals Belong Inline In Markdown" in dashboard_content
    assert "Primary markdown homes" in dashboard_content
    assert "Core inline" in dashboard_content
    assert "Stage-Local Artifact Gallery" in dashboard_content
    assert "fig_actual_vs_predicted.png" in dashboard_content
