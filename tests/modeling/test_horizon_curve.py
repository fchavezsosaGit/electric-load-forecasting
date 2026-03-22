"""Unit tests for Stage-8 horizon-curve registry reuse behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    """Load the Stage-8 module directly from disk for isolated unit testing."""
    path = Path("scripts/modeling/horizon_curve.py").resolve()
    spec = importlib.util.spec_from_file_location("test_horizon_curve_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_horizon_curve_reuses_existing_challenger_sweep_registry(tmp_path, monkeypatch):
    """Reuse a comparable saved sweep instead of rerunning Stage-7 unnecessarily."""
    module = _load_module()

    recommended_candidate_path = tmp_path / "recommended_candidate.json"
    recommended_candidate_path.write_text("{}", encoding="utf-8")

    candidate_results_path = tmp_path / "candidate_results.csv"
    pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "candidate_label": "hgb-balanced::persistence_raw_blend_e35",
                "resolution": "10min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
                "requested_origin_policy": "uniform",
                "run_id": "20260310T023050847923Z",
                "run_path": "outputs/007_rollout/commercial_facility/20260310T023050847923Z",
                "selection_metric_value": 622.391884,
                "selection_metric_pct": 31.094290,
                "endpoint_mae": 592.099063,
                "endpoint_mae_pct": 30.364591,
                "path_mae": 622.391884,
                "path_mae_pct": 31.094290,
                "phase_mean_mae": 510.0,
                "phase_mean_mae_pct": 25.0,
                "next_lock_mae": 515.0,
                "next_lock_mae_pct": 24.8,
                "profile_shape_mae": 540.0,
                "profile_shape_mae_pct": 26.2,
                "energy_mae": 120.0,
                "energy_mae_pct": 5.8,
                "mean_coverage": 1.0,
                "origin_n": 8,
            }
        ]
    ).to_csv(candidate_results_path, index=False)

    registry = pd.DataFrame(
        [
            {
                "sweep_run_id": "20260310T023050758459Z",
                "generated_at_utc": pd.Timestamp("2026-03-10T02:37:35Z"),
                "requested_horizon_minutes": 240,
                "selection_target": "path_mae",
                "recommended_origin_policy": "uniform",
                "recommended_candidate_label": "hgb-balanced::persistence_raw_blend_e35",
                "recommended_resolution": "10min",
                "recommended_feature_set": "minimal",
                "recommended_model_label": "hgb-balanced",
                "recommended_candidate_path": str(recommended_candidate_path),
                "candidate_results_path": str(candidate_results_path),
                "endpoint_mae": 592.099063,
                "endpoint_mae_pct": 30.364591,
                "path_mae": 622.391884,
                "path_mae_pct": 31.094290,
                "phase_mean_mae": 510.0,
                "phase_mean_mae_pct": 25.0,
                "next_lock_mae": 515.0,
                "next_lock_mae_pct": 24.8,
                "profile_shape_mae": 540.0,
                "profile_shape_mae_pct": 26.2,
                "energy_mae": 120.0,
                "energy_mae_pct": 5.8,
                "persistence_endpoint_mae": 884.654476,
                "persistence_endpoint_mae_pct": 45.0,
                "persistence_path_mae": 711.377851,
                "persistence_path_mae_pct": 35.0,
                "persistence_phase_mean_mae": 560.0,
                "persistence_phase_mean_mae_pct": 27.0,
                "persistence_next_lock_mae": 540.0,
                "persistence_next_lock_mae_pct": 26.0,
                "persistence_profile_shape_mae": 620.0,
                "persistence_profile_shape_mae_pct": 30.0,
                "best_baseline_endpoint_label": "avg_workday",
                "best_baseline_endpoint_mae": 750.888084,
                "best_baseline_endpoint_mae_pct": 38.0,
                "best_baseline_path_label": "persistence",
                "best_baseline_path_mae": 711.377851,
                "best_baseline_path_mae_pct": 35.0,
                "best_baseline_phase_label": "persistence",
                "best_baseline_phase_mae": 560.0,
                "best_baseline_phase_mae_pct": 27.0,
                "best_baseline_next_lock_label": "persistence",
                "best_baseline_next_lock_mae": 540.0,
                "best_baseline_next_lock_mae_pct": 26.0,
                "best_baseline_profile_shape_label": "avg_workday",
                "best_baseline_profile_shape_mae": 600.0,
                "best_baseline_profile_shape_mae_pct": 29.0,
                "beats_persistence_endpoint": True,
                "beats_persistence_path": True,
                "beats_persistence_phase": True,
                "beats_persistence_next_lock": True,
                "beats_persistence_profile_shape": True,
                "beats_best_baseline_endpoint": True,
                "beats_best_baseline_path": True,
                "beats_best_baseline_phase": True,
                "beats_best_baseline_next_lock": True,
                "beats_best_baseline_profile_shape": True,
                "origin_n": 8,
                "mean_coverage": 1.0,
            }
        ]
    )

    monkeypatch.setattr(module, "_build_challenger_sweep_registry_snapshot", lambda _root: registry)
    monkeypatch.setattr(
        module,
        "run_rollout_challenger_sweep",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("Stage-8 should reuse the registry row")),
    )

    result = module.run_horizon_curve(
        output_root=tmp_path / "curve",
        horizons_minutes=[240],
        origins=8,
        origin_policy="uniform",
        selection_target="path_mae",
        max_candidates=8,
        include_stage5_anchor=False,
        reuse_existing_sweeps=True,
    )

    summary = result["summary"]
    assert summary["horizon_minutes"].tolist() == [240]
    assert summary.iloc[0]["selection_policy"] == "challenger_sweep_registry"
    assert summary.iloc[0]["candidate_label"] == "hgb-balanced::persistence_raw_blend_e35"
    assert bool(summary.iloc[0]["beats_persistence_path"]) is True
    assert float(summary.iloc[0]["learned_profile_shape_mae"]) == 540.0


def test_run_horizon_curve_reruns_when_registry_origin_policy_mismatches(tmp_path, monkeypatch):
    """Rerun the sweep when saved evidence is not comparable to the requested policy."""
    module = _load_module()

    candidate_results_path = tmp_path / "candidate_results.csv"
    pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "candidate_label": "old-midnight-candidate",
                "resolution": "1min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
                "requested_origin_policy": "midnight",
                "run_id": "20260310T000623233402Z",
                "run_path": "outputs/007_rollout/commercial_facility/20260310T000623233402Z",
                "selection_metric_value": 361.418785,
                "selection_metric_pct": 14.830783,
                "endpoint_mae": 468.35,
                "endpoint_mae_pct": 18.441639,
                "path_mae": 361.418785,
                "path_mae_pct": 14.830783,
                "phase_mean_mae": 240.0,
                "phase_mean_mae_pct": 9.8,
                "next_lock_mae": 240.0,
                "next_lock_mae_pct": 9.8,
                "profile_shape_mae": 260.0,
                "profile_shape_mae_pct": 10.6,
                "energy_mae": 60.0,
                "energy_mae_pct": 2.4,
                "mean_coverage": 1.0,
                "origin_n": 2,
            }
        ]
    ).to_csv(candidate_results_path, index=False)

    registry = pd.DataFrame(
        [
            {
                "sweep_run_id": "20260310T000605523800Z",
                "generated_at_utc": pd.Timestamp("2026-03-10T00:06:26Z"),
                "requested_horizon_minutes": 15,
                "selection_target": "path_mae",
                "recommended_origin_policy": "midnight",
                "recommended_candidate_label": "old-midnight-candidate",
                "recommended_resolution": "1min",
                "recommended_feature_set": "minimal",
                "recommended_model_label": "hgb-balanced",
                "candidate_results_path": str(candidate_results_path),
                "recommended_candidate_path": str(tmp_path / "recommended_candidate.json"),
                "endpoint_mae": 468.35,
                "endpoint_mae_pct": 18.441639,
                "path_mae": 361.418785,
                "path_mae_pct": 14.830783,
                "phase_mean_mae": 240.0,
                "phase_mean_mae_pct": 9.8,
                "next_lock_mae": 240.0,
                "next_lock_mae_pct": 9.8,
                "profile_shape_mae": 260.0,
                "profile_shape_mae_pct": 10.6,
                "energy_mae": 60.0,
                "energy_mae_pct": 2.4,
                "persistence_endpoint_mae": 606.741667,
                "persistence_endpoint_mae_pct": 23.0,
                "persistence_path_mae": 412.158286,
                "persistence_path_mae_pct": 16.0,
                "persistence_phase_mean_mae": 260.0,
                "persistence_phase_mean_mae_pct": 10.4,
                "persistence_next_lock_mae": 260.0,
                "persistence_next_lock_mae_pct": 10.4,
                "persistence_profile_shape_mae": 280.0,
                "persistence_profile_shape_mae_pct": 11.1,
                "best_baseline_endpoint_label": "avg_workday",
                "best_baseline_endpoint_mae": 468.35,
                "best_baseline_endpoint_mae_pct": 18.4,
                "best_baseline_path_label": "persistence",
                "best_baseline_path_mae": 412.158286,
                "best_baseline_path_mae_pct": 16.0,
                "best_baseline_phase_label": "persistence",
                "best_baseline_phase_mae": 260.0,
                "best_baseline_phase_mae_pct": 10.4,
                "best_baseline_next_lock_label": "persistence",
                "best_baseline_next_lock_mae": 260.0,
                "best_baseline_next_lock_mae_pct": 10.4,
                "best_baseline_profile_shape_label": "persistence",
                "best_baseline_profile_shape_mae": 280.0,
                "best_baseline_profile_shape_mae_pct": 11.1,
                "beats_persistence_endpoint": True,
                "beats_persistence_path": True,
                "beats_persistence_phase": True,
                "beats_persistence_next_lock": True,
                "beats_persistence_profile_shape": True,
                "beats_best_baseline_endpoint": False,
                "beats_best_baseline_path": True,
                "beats_best_baseline_phase": True,
                "beats_best_baseline_next_lock": True,
                "beats_best_baseline_profile_shape": True,
                "origin_n": 2,
                "mean_coverage": 1.0,
            }
        ]
    )

    called: dict[str, object] = {}

    def _fake_run_rollout_challenger_sweep(**kwargs):
        called.update(kwargs)
        sweep_dir = tmp_path / "rerun_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        return {
            "sweep_dir": sweep_dir,
            "recommended": {
                "recommended_origin_policy": "uniform",
                "candidate_label": "uniform-rerun-candidate",
                "resolution": "1min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
                "endpoint_mae": 470.0,
                "endpoint_mae_pct": 19.0,
                "path_mae": 365.0,
                "path_mae_pct": 15.0,
                "phase_mean_mae": 245.0,
                "phase_mean_mae_pct": 10.0,
                "next_lock_mae": 245.0,
                "next_lock_mae_pct": 10.0,
                "profile_shape_mae": 268.0,
                "profile_shape_mae_pct": 10.8,
                "energy_mae": 62.0,
                "energy_mae_pct": 2.5,
                "persistence_endpoint_mae": 606.741667,
                "persistence_endpoint_mae_pct": 23.0,
                "persistence_path_mae": 412.158286,
                "persistence_path_mae_pct": 16.0,
                "persistence_phase_mean_mae": 260.0,
                "persistence_phase_mean_mae_pct": 10.4,
                "persistence_next_lock_mae": 260.0,
                "persistence_next_lock_mae_pct": 10.4,
                "persistence_profile_shape_mae": 280.0,
                "persistence_profile_shape_mae_pct": 11.1,
                "best_baseline_endpoint_label": "avg_workday",
                "best_baseline_endpoint_mae": 468.35,
                "best_baseline_endpoint_mae_pct": 18.4,
                "best_baseline_path_label": "persistence",
                "best_baseline_path_mae": 412.158286,
                "best_baseline_path_mae_pct": 16.0,
                "best_baseline_phase_label": "persistence",
                "best_baseline_phase_mae": 260.0,
                "best_baseline_phase_mae_pct": 10.4,
                "best_baseline_next_lock_label": "persistence",
                "best_baseline_next_lock_mae": 260.0,
                "best_baseline_next_lock_mae_pct": 10.4,
                "best_baseline_profile_shape_label": "persistence",
                "best_baseline_profile_shape_mae": 280.0,
                "best_baseline_profile_shape_mae_pct": 11.1,
                "beats_persistence_endpoint": True,
                "beats_persistence_path": True,
                "beats_persistence_phase": True,
                "beats_persistence_next_lock": True,
                "beats_persistence_profile_shape": True,
                "beats_best_baseline_endpoint": False,
                "beats_best_baseline_path": True,
                "beats_best_baseline_phase": True,
                "beats_best_baseline_next_lock": True,
                "beats_best_baseline_profile_shape": True,
                "origin_n": 8,
            },
            "candidate_results": pd.DataFrame(
                [
                    {
                        "candidate_rank": 1,
                        "candidate_label": "uniform-rerun-candidate",
                        "resolution": "1min",
                        "feature_set": "minimal",
                        "model_label": "hgb-balanced",
                        "requested_origin_policy": "uniform",
                        "run_id": "20260310T999999999999Z",
                        "run_path": "outputs/007_rollout/commercial_facility/20260310T999999999999Z",
                        "selection_metric_value": 365.0,
                        "selection_metric_pct": 15.0,
                        "endpoint_mae": 470.0,
                        "endpoint_mae_pct": 19.0,
                        "path_mae": 365.0,
                        "path_mae_pct": 15.0,
                        "phase_mean_mae": 245.0,
                        "phase_mean_mae_pct": 10.0,
                        "next_lock_mae": 245.0,
                        "next_lock_mae_pct": 10.0,
                        "profile_shape_mae": 268.0,
                        "profile_shape_mae_pct": 10.8,
                        "energy_mae": 62.0,
                        "energy_mae_pct": 2.5,
                        "mean_coverage": 1.0,
                        "origin_n": 8,
                    }
                ]
            ),
        }

    monkeypatch.setattr(module, "_build_challenger_sweep_registry_snapshot", lambda _root: registry)
    monkeypatch.setattr(module, "run_rollout_challenger_sweep", _fake_run_rollout_challenger_sweep)

    result = module.run_horizon_curve(
        output_root=tmp_path / "curve",
        horizons_minutes=[15],
        origins=8,
        origin_policy="uniform",
        selection_target="path_mae",
        max_candidates=8,
        include_stage5_anchor=False,
        reuse_existing_sweeps=True,
    )

    assert called["origin_policy"] == "uniform"
    summary = result["summary"]
    assert summary.iloc[0]["selection_policy"] == "rollout_challenger_sweep"
    assert summary.iloc[0]["candidate_label"] == "uniform-rerun-candidate"
