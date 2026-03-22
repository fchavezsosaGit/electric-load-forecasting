"""Unit tests for Stage-6 winner selection logic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_multires_compare_module():
    """Load the Stage-6 implementation module for winner-selection tests."""
    path = Path("scripts/modeling/multires_compare.py").resolve()
    spec = importlib.util.spec_from_file_location("test_multires_compare_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load multires_compare module for testing.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_winners_can_choose_non_persistence_baseline():
    """Allow a non-persistence baseline to win when learned candidates do not clear gates."""
    module = _load_multires_compare_module()
    summary = pd.DataFrame(
        [
            {
                "comparison_mode": "matched_horizon",
                "resolution": "10min",
                "horizon_minutes": 60,
                "feature_set": "baseline",
                "model_label": "persistence",
                "baseline_label": "persistence",
                "candidate_type": "baseline",
                "forecast_strategy": "path_baseline",
                "mae": 100.0,
                "rmse": 120.0,
                "mae_pct": 20.0,
                "path_mae": 100.0,
                "path_mae_pct": 20.0,
                "mae_ratio_to_persistence": 1.0,
                "rmse_ratio_to_persistence": 1.0,
                "fold_std_mae_ratio": 0.0,
                "n_eval": 10,
                "eval_coverage": 1.0,
                "runtime_seconds": 0.0,
                "fold_n": 4,
                "source_mode": "bronze_direct",
            },
            {
                "comparison_mode": "matched_horizon",
                "resolution": "10min",
                "horizon_minutes": 60,
                "feature_set": "baseline",
                "model_label": "anchored_workday",
                "baseline_label": "persistence",
                "candidate_type": "baseline",
                "forecast_strategy": "path_baseline",
                "mae": 92.0,
                "rmse": 110.0,
                "mae_pct": 18.4,
                "path_mae": 91.0,
                "path_mae_pct": 18.2,
                "mae_ratio_to_persistence": 0.92,
                "rmse_ratio_to_persistence": 0.91,
                "fold_std_mae_ratio": 0.08,
                "n_eval": 10,
                "eval_coverage": 1.0,
                "runtime_seconds": 0.0,
                "fold_n": 4,
                "source_mode": "bronze_direct",
            },
            {
                "comparison_mode": "matched_horizon",
                "resolution": "10min",
                "horizon_minutes": 60,
                "feature_set": "curated",
                "model_label": "hgb-balanced",
                "baseline_label": "persistence",
                "candidate_type": "learned",
                "forecast_strategy": "direct_endpoint",
                "mae": 96.0,
                "rmse": 100.0,
                "mae_pct": 19.2,
                "path_mae": float("nan"),
                "path_mae_pct": float("nan"),
                "mae_ratio_to_persistence": 0.96,
                "rmse_ratio_to_persistence": 0.83,
                "fold_std_mae_ratio": 0.35,
                "n_eval": 10,
                "eval_coverage": 1.0,
                "runtime_seconds": 2.0,
                "fold_n": 4,
                "source_mode": "bronze_direct",
            },
        ]
    )

    winners = module._select_winners(summary)

    assert winners.iloc[0]["winner_type"] == "baseline_model"
    assert winners.iloc[0]["winner_model_label"] == "anchored_workday"
    assert winners.iloc[0]["winner_forecast_strategy"] == "path_baseline"
    assert winners.iloc[0]["winner_endpoint_mae"] == 92.0
    assert winners.iloc[0]["winner_endpoint_mae_pct"] == 18.4
    assert winners.iloc[0]["winner_path_mae"] == 91.0
    assert winners.iloc[0]["winner_path_mae_pct"] == 18.2


def test_build_winner_registry_backfills_metrics_from_matched_horizon_rows(tmp_path):
    """Backfill missing winner metrics from matched-horizon rows when summaries omit them."""
    module = _load_multires_compare_module()
    output_root = tmp_path / "outputs" / "006_multires" / "commercial_facility"
    run_dir = output_root / "20260309T000000000000Z"
    run_dir.mkdir(parents=True)

    (run_dir / "run_manifest.json").write_text(
        '{"generated_at_utc":"2026-03-09T00:00:00+00:00","mode":"focus_60m","comparison_mode":"matched_horizon"}',
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "use_case": "matched_horizon_60m",
                "winner_type": "learned_model",
                "winner_resolution": "5min",
                "winner_feature_set": "minimal",
                "winner_model_label": "hgb-balanced",
                "winner_forecast_strategy": "recursive",
                "winner_horizon_minutes": 60,
                "decision_reason": "test",
                "practical_gain_passed": True,
                "pareto_passed": True,
            }
        ]
    ).to_csv(run_dir / "selection_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "comparison_mode": "matched_horizon",
                "resolution": "5min",
                "horizon_minutes": 60,
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
                "forecast_strategy": "recursive",
                "mae": 1148.166851,
                "mae_pct": 42.691558,
                "path_mae": 1151.446627,
                "path_mae_pct": 36.61195,
                "mae_ratio_to_persistence": 0.654046,
                "rmse_ratio_to_persistence": 0.672733,
                "eval_coverage": 1.0,
                "runtime_seconds": 0.807373,
            }
        ]
    ).to_csv(run_dir / "matched_horizon_metrics.csv", index=False)

    registry = module._build_winner_registry_snapshot(output_root)

    assert registry.iloc[0]["winner_endpoint_mae"] == 1148.166851
    assert registry.iloc[0]["winner_endpoint_mae_pct"] == 42.691558
    assert registry.iloc[0]["winner_path_mae"] == 1151.446627
    assert registry.iloc[0]["winner_path_mae_pct"] == 36.61195
