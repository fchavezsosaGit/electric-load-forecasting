"""Unit tests for the Stage-10 forecast-control backtest helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.modeling import forecast_control_backtest as module


def test_extract_candidate_path_forward_and_back_fills_native_forecast_steps():
    """Native rollout steps should expand onto the minute grid without leaving gaps."""
    origin = pd.Timestamp("2026-03-01T00:00:00")
    detail = pd.DataFrame(
        [
            {
                "origin_timestamp": origin.isoformat(),
                "candidate_label": "model::raw",
                "forecast_timestamp": (origin + pd.Timedelta(minutes=5)).isoformat(),
                "predicted_load": 100.0,
            },
            {
                "origin_timestamp": origin.isoformat(),
                "candidate_label": "model::raw",
                "forecast_timestamp": (origin + pd.Timedelta(minutes=10)).isoformat(),
                "predicted_load": 200.0,
            },
        ]
    )
    minute_index = pd.date_range(origin + pd.Timedelta(minutes=1), periods=10, freq="1min")

    series = module._extract_candidate_path(
        detail,
        origin_timestamp=origin,
        candidate_label="model::raw",
        minute_index=minute_index,
    )

    assert float(series.iloc[0]) == 100.0
    assert float(series.loc[origin + pd.Timedelta(minutes=5)]) == 100.0
    assert float(series.loc[origin + pd.Timedelta(minutes=10)]) == 200.0


def test_extract_candidate_path_deduplicates_duplicate_forecast_timestamps():
    """Duplicate detailed rows for the same forecast timestamp should collapse to the last value."""
    origin = pd.Timestamp("2026-03-01T00:00:00")
    detail = pd.DataFrame(
        [
            {
                "origin_timestamp": origin.isoformat(),
                "candidate_label": "model::raw",
                "forecast_timestamp": (origin + pd.Timedelta(minutes=5)).isoformat(),
                "predicted_load": 100.0,
            },
            {
                "origin_timestamp": origin.isoformat(),
                "candidate_label": "model::raw",
                "forecast_timestamp": (origin + pd.Timedelta(minutes=5)).isoformat(),
                "predicted_load": 120.0,
            },
            {
                "origin_timestamp": origin.isoformat(),
                "candidate_label": "model::raw",
                "forecast_timestamp": (origin + pd.Timedelta(minutes=10)).isoformat(),
                "predicted_load": 200.0,
            },
        ]
    )
    minute_index = pd.date_range(origin + pd.Timedelta(minutes=1), periods=10, freq="1min")

    series = module._extract_candidate_path(
        detail,
        origin_timestamp=origin,
        candidate_label="model::raw",
        minute_index=minute_index,
    )

    assert float(series.loc[origin + pd.Timedelta(minutes=5)]) == 120.0


def test_apply_rollout_updates_only_overwrites_the_requested_update_window():
    """Layer updates should replace only the horizon segment after each origin."""
    origin = pd.Timestamp("2026-03-01T00:00:00")
    minute_index = pd.date_range(origin + pd.Timedelta(minutes=1), periods=8, freq="1min")
    base = pd.Series([10.0] * len(minute_index), index=minute_index)
    detail = pd.DataFrame(
        [
            {
                "origin_timestamp": origin.isoformat(),
                "candidate_label": "model::raw",
                "forecast_timestamp": timestamp.isoformat(),
                "predicted_load": 20.0,
            }
            for timestamp in minute_index[:4]
        ]
    )

    updated = module._apply_rollout_updates(
        base,
        detail_by_origin=detail,
        candidate_label="model::raw",
        update_origins=[origin],
        horizon_minutes=4,
    )

    assert all(float(value) == 20.0 for value in updated.iloc[:4])
    assert all(float(value) == 10.0 for value in updated.iloc[4:])


def test_apply_nowcast_updates_only_overwrites_timestamps_with_exact_predictions():
    """Minute-level nowcasts should replace only the timestamps they explicitly cover."""
    minute_index = pd.date_range("2026-03-01T00:01:00", periods=5, freq="1min")
    base = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0], index=minute_index, dtype=float)
    nowcast = pd.Series(
        [20.0, 25.0],
        index=pd.to_datetime(["2026-03-01T00:02:00", "2026-03-01T00:04:00"]),
        dtype=float,
    )

    updated = module._apply_nowcast_updates(base, nowcast)

    assert float(updated.loc[pd.Timestamp("2026-03-01T00:01:00")]) == 10.0
    assert float(updated.loc[pd.Timestamp("2026-03-01T00:02:00")]) == 20.0
    assert float(updated.loc[pd.Timestamp("2026-03-01T00:03:00")]) == 10.0
    assert float(updated.loc[pd.Timestamp("2026-03-01T00:04:00")]) == 25.0


def test_load_stage5_nowcast_anchor_prefers_best_registry_winner(monkeypatch, tmp_path):
    """Use the strongest cross-run Stage-5 holdout winner instead of only `latest/`."""
    performance_root = tmp_path / "outputs" / "005_performance"
    performance_root.mkdir(parents=True, exist_ok=True)
    (performance_root / "holdout_registry.csv").write_text(
        "\n".join(
            [
                "run_id,generated_at_utc,mode,resolution,learned_candidate_label,learned_feature_set,learned_model_label,learned_target_mode,learned_mae,learned_mae_pct,learned_mae_ratio_to_persistence,persistence_mae,persistence_mae_pct,learned_beats_persistence,recommended_candidate_label,recommended_candidate_type,decision_reason,holdout_evaluation_artifact,deployment_recommendation_artifact,run_manifest_artifact,blend_window,blend_sharpness,blend_min_weight,blend_max_weight",
                "older-full,2026-03-09T21:32:40+00:00,full,1min,full_stable/hgb-frontier-lr010-l2001/raw+blend,full_stable,hgb-frontier-lr010-l2001,raw+blend,163.0,7.86,0.938,173.7,8.38,True,full_stable/hgb-frontier-lr010-l2001/raw+blend,promoted_learned,beat persistence,outputs/005_performance/commercial_facility/older-full/holdout_evaluation.csv,outputs/005_performance/commercial_facility/older-full/deployment_recommendation.json,outputs/005_performance/commercial_facility/older-full/run_manifest.json,240,9.0,0.0,1.0",
                "latest-quick,2026-03-11T11:41:23+00:00,quick,1min,curated_ramp/hgb-balanced/residual+blend,curated_ramp,hgb-balanced,residual+blend,174.9,8.44,1.007,173.7,8.38,False,persistence,baseline,did not beat persistence,outputs/005_performance/commercial_facility/latest-quick/holdout_evaluation.csv,outputs/005_performance/commercial_facility/latest-quick/deployment_recommendation.json,outputs/005_performance/commercial_facility/latest-quick/run_manifest.json,60,9.0,0.0,1.0",
            ]
        ),
        encoding="utf-8",
    )
    latest_dir = performance_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "run_manifest.json").write_text('{"run_id":"latest-quick"}', encoding="utf-8")
    monkeypatch.setattr(module, "preferred_output_path", lambda _path: performance_root)

    anchor = module._load_stage5_nowcast_anchor()

    assert str(anchor["candidate_label"]) == "full_stable/hgb-frontier-lr010-l2001/raw+blend"
    assert bool(anchor["beats_persistence"]) is True
    assert float(anchor["blend_window"]) == 240.0
    assert str(anchor["source_run_id"]) == "older-full"


def test_load_stage5_dynamic_model_catalog_reads_latest_manifest(monkeypatch, tmp_path):
    """Hydrate adaptive Stage-5 HGB specs from the latest manifest for Stage-10 replay."""
    performance_root = tmp_path / "outputs" / "005_performance"
    latest_dir = performance_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_label": "hgb-auto-custom",
                        "family": "hgb",
                        "params": {
                            "max_depth": 7,
                            "max_iter": 300,
                            "learning_rate": 0.05,
                            "min_samples_leaf": 100,
                            "l2_regularization": 0.0,
                            "early_stopping": False,
                            "random_state": 42,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(module.PATHS, "outputs_performance_dir", str(performance_root))
    monkeypatch.setattr(module, "preferred_output_path", lambda path: Path(path))

    catalog = module._load_stage5_dynamic_model_catalog()

    assert "hgb-auto-custom" in catalog
    assert catalog["hgb-auto-custom"].family == "hgb"


def test_load_stage5_nowcast_candidate_pool_includes_registry_rows_before_latest_scoreboard(
    monkeypatch, tmp_path
):
    """Prior holdout-backed winners should enter the exact-control pool before latest-only rows."""
    performance_root = tmp_path / "outputs" / "005_performance"
    latest_dir = performance_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (performance_root / "holdout_registry.csv").write_text(
        "\n".join(
            [
                "run_id,generated_at_utc,mode,resolution,learned_candidate_label,learned_feature_set,learned_model_label,learned_target_mode,learned_mae,learned_mae_pct,learned_mae_ratio_to_persistence,persistence_mae,persistence_mae_pct,learned_beats_persistence,recommended_candidate_label,recommended_candidate_type,decision_reason,holdout_evaluation_artifact,deployment_recommendation_artifact,run_manifest_artifact,blend_window,blend_sharpness,blend_min_weight,blend_max_weight",
                "older-full,2026-03-09T21:32:40+00:00,full,1min,full_stable/hgb-frontier-lr010-l2001/raw+blend,full_stable,hgb-frontier-lr010-l2001,raw+blend,163.0,7.86,0.938,173.7,8.38,True,full_stable/hgb-frontier-lr010-l2001/raw+blend,promoted_learned,beat persistence,outputs/005_performance/commercial_facility/older-full/holdout_evaluation.csv,outputs/005_performance/commercial_facility/older-full/deployment_recommendation.json,outputs/005_performance/commercial_facility/older-full/run_manifest.json,240,9.0,0.0,1.0",
            ]
        ),
        encoding="utf-8",
    )
    (latest_dir / "selection_scoreboard.csv").write_text(
        "\n".join(
            [
                "resolution,feature_set,model_label,target_mode,fold_mean_mae_ratio,fold_std_mae_ratio,fold_n,raw_validate_mae,raw_validate_mae_pct,raw_validate_rmse_pct,mean_coverage",
                "1min,curated_ramp,hgb-balanced,residual+blend,0.94,0.02,2,586.0,14.68,28.98,0.99",
            ]
        ),
        encoding="utf-8",
    )
    (latest_dir / "run_manifest.json").write_text('{"run_id":"latest-quick"}', encoding="utf-8")
    monkeypatch.setattr(module, "preferred_output_path", lambda _path: performance_root)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "nowcast_candidate_pool_size", 4)

    pool = module._load_stage5_nowcast_candidate_pool(
        upstream_anchor={
            "candidate_label": "persistence",
            "candidate_type": "baseline",
            "resolution": "1min",
            "feature_set": "baseline",
            "model_label": "persistence",
            "target_mode": "baseline",
        }
    )

    assert pool[0]["candidate_label"] == "persistence"
    assert pool[1]["candidate_label"] == "full_stable/hgb-frontier-lr010-l2001/raw+blend"
    assert float(pool[1]["blend_window"]) == 240.0
    assert pool[2]["candidate_label"] == "curated_ramp/hgb-balanced/residual+blend"


def test_load_stage5_nowcast_candidate_pool_includes_blend_finalists_with_saved_configs(
    monkeypatch, tmp_path
):
    """Stage-10 should consume Stage-5 blend finalists before falling back to raw scoreboard rows."""
    performance_root = tmp_path / "outputs" / "005_performance"
    latest_dir = performance_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (performance_root / "holdout_registry.csv").write_text(
        "\n".join(
            [
                "run_id,generated_at_utc,mode,resolution,learned_candidate_label,learned_feature_set,learned_model_label,learned_target_mode,learned_mae,learned_mae_pct,learned_mae_ratio_to_persistence,persistence_mae,persistence_mae_pct,learned_beats_persistence,recommended_candidate_label,recommended_candidate_type,decision_reason,holdout_evaluation_artifact,deployment_recommendation_artifact,run_manifest_artifact,blend_window,blend_sharpness,blend_min_weight,blend_max_weight",
                "older-full,2026-03-09T21:32:40+00:00,full,1min,full_stable/hgb-frontier-lr010-l2001/raw+blend,full_stable,hgb-frontier-lr010-l2001,raw+blend,163.0,7.86,0.938,173.7,8.38,True,full_stable/hgb-frontier-lr010-l2001/raw+blend,promoted_learned,beat persistence,outputs/005_performance/commercial_facility/older-full/holdout_evaluation.csv,outputs/005_performance/commercial_facility/older-full/deployment_recommendation.json,outputs/005_performance/commercial_facility/older-full/run_manifest.json,240,9.0,0.0,1.0",
            ]
        ),
        encoding="utf-8",
    )
    (latest_dir / "blend_finalists.csv").write_text(
        "\n".join(
            [
                "resolution,feature_set,model_label,source_target_mode,target_mode,candidate_label,fold_mean_mae_ratio,fold_std_mae_ratio,blend_validate_mae,blend_validate_mae_pct,blend_validate_rmse_pct,mean_coverage,max_fold_degrade_pct,selected_blend_window,selected_blend_sharpness,selected_blend_min_weight,selected_blend_max_weight,meets_p2_fold_degrade_cap,preferred_candidate_match,blend_rank",
                "1min,minimal_phase_anchor,hgb-frontier-lr010-leaf100,raw,raw+blend,minimal_phase_anchor/hgb-frontier-lr010-leaf100/raw+blend,0.991,0.015,612.0,15.3,29.7,0.995,0.8,240,12.0,0.0,0.6,True,False,1",
            ]
        ),
        encoding="utf-8",
    )
    (latest_dir / "selection_scoreboard.csv").write_text(
        "\n".join(
            [
                "resolution,feature_set,model_label,target_mode,fold_mean_mae_ratio,fold_std_mae_ratio,fold_n,raw_validate_mae,raw_validate_mae_pct,raw_validate_rmse_pct,mean_coverage",
                "1min,minimal_phase_anchor,hgb-frontier-lr010-leaf100,raw+blend,0.991,0.015,2,612.0,15.3,29.7,0.995",
                "1min,minimal_phase,hgb-balanced,residual,1.020,0.030,2,640.0,16.0,30.0,0.995",
            ]
        ),
        encoding="utf-8",
    )
    (latest_dir / "run_manifest.json").write_text('{"run_id":"latest-quick"}', encoding="utf-8")
    monkeypatch.setattr(module, "preferred_output_path", lambda _path: performance_root)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "nowcast_candidate_pool_size", 4)

    pool = module._load_stage5_nowcast_candidate_pool(
        upstream_anchor={
            "candidate_label": "persistence",
            "candidate_type": "baseline",
            "resolution": "1min",
            "feature_set": "baseline",
            "model_label": "persistence",
            "target_mode": "baseline",
        }
    )

    assert pool[0]["candidate_label"] == "persistence"
    assert pool[1]["candidate_label"] == "full_stable/hgb-frontier-lr010-l2001/raw+blend"
    assert pool[2]["candidate_label"] == "minimal_phase_anchor/hgb-frontier-lr010-leaf100/raw+blend"
    assert float(pool[2]["blend_sharpness"]) == 12.0
    assert str(pool[2]["pool_source_type"]) == "stage5_blend_finalists"
    assert pool[3]["candidate_label"] == "minimal_phase/hgb-balanced/residual"


def test_load_stage5_nowcast_candidate_pool_diversifies_blend_finalists_by_feature_set(
    monkeypatch, tmp_path
):
    """Blend finalists should reserve pool space for different Stage-5 feature families."""
    performance_root = tmp_path / "outputs" / "005_performance"
    latest_dir = performance_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (performance_root / "holdout_registry.csv").write_text(
        "\n".join(
            [
                "run_id,generated_at_utc,mode,resolution,learned_candidate_label,learned_feature_set,learned_model_label,learned_target_mode,learned_mae,learned_mae_pct,learned_mae_ratio_to_persistence,persistence_mae,persistence_mae_pct,learned_beats_persistence,recommended_candidate_label,recommended_candidate_type,decision_reason,holdout_evaluation_artifact,deployment_recommendation_artifact,run_manifest_artifact,blend_window,blend_sharpness,blend_min_weight,blend_max_weight",
                "older-full,2026-03-09T21:32:40+00:00,full,1min,full_stable/hgb-frontier-lr010-l2001/raw+blend,full_stable,hgb-frontier-lr010-l2001,raw+blend,163.0,7.86,0.938,173.7,8.38,True,full_stable/hgb-frontier-lr010-l2001/raw+blend,promoted_learned,beat persistence,outputs/005_performance/commercial_facility/older-full/holdout_evaluation.csv,outputs/005_performance/commercial_facility/older-full/deployment_recommendation.json,outputs/005_performance/commercial_facility/older-full/run_manifest.json,240,9.0,0.0,1.0",
                "new-quick,2026-03-12T05:13:39+00:00,quick,1min,curated_ramp/hgb-frontier-lr010-leaf100/residual+blend,curated_ramp,hgb-frontier-lr010-leaf100,residual+blend,174.9,8.44,1.007,173.7,8.38,False,persistence,baseline,did not beat persistence,outputs/005_performance/commercial_facility/new-quick/holdout_evaluation.csv,outputs/005_performance/commercial_facility/new-quick/deployment_recommendation.json,outputs/005_performance/commercial_facility/new-quick/run_manifest.json,60,9.0,0.0,1.0",
            ]
        ),
        encoding="utf-8",
    )
    (latest_dir / "blend_finalists.csv").write_text(
        "\n".join(
            [
                "resolution,feature_set,model_label,source_target_mode,target_mode,candidate_label,fold_mean_mae_ratio,fold_std_mae_ratio,blend_validate_mae,blend_validate_mae_pct,blend_validate_rmse_pct,mean_coverage,max_fold_degrade_pct,selected_blend_window,selected_blend_sharpness,selected_blend_min_weight,selected_blend_max_weight,meets_p2_fold_degrade_cap,preferred_candidate_match,blend_rank",
                "1min,curated_ramp,hgb-balanced,residual,residual+blend,curated_ramp/hgb-balanced/residual+blend,0.944,0.020,586.0,14.7,29.0,0.995,-3.3,60,9.0,0.0,1.0,True,False,1",
                "1min,full_stable,hgb-balanced,raw,raw+blend,full_stable/hgb-balanced/raw+blend,0.950,0.023,589.0,14.8,29.0,0.995,-6.7,60,9.0,0.0,1.0,True,False,2",
                "1min,minimal_phase_anchor,hgb-frontier-lr010-leaf100,raw,raw+blend,minimal_phase_anchor/hgb-frontier-lr010-leaf100/raw+blend,0.987,0.001,613.0,15.4,30.2,0.995,-10.0,120,9.0,0.05,0.95,True,False,3",
                "1min,minimal_phase,hgb-balanced,raw,raw+blend,minimal_phase/hgb-balanced/raw+blend,0.989,0.001,614.0,15.4,30.3,0.995,-11.0,120,9.0,0.05,0.95,True,False,4",
            ]
        ),
        encoding="utf-8",
    )
    (latest_dir / "selection_scoreboard.csv").write_text(
        "\n".join(
            [
                "resolution,feature_set,model_label,target_mode,fold_mean_mae_ratio,fold_std_mae_ratio,fold_n,raw_validate_mae,raw_validate_mae_pct,raw_validate_rmse_pct,mean_coverage",
                "1min,curated_ramp,hgb-balanced,residual,1.00,0.03,2,620.0,15.5,29.5,0.995",
            ]
        ),
        encoding="utf-8",
    )
    (latest_dir / "run_manifest.json").write_text('{"run_id":"latest-quick"}', encoding="utf-8")
    monkeypatch.setattr(module, "preferred_output_path", lambda _path: performance_root)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "nowcast_candidate_pool_size", 5)

    pool = module._load_stage5_nowcast_candidate_pool(
        upstream_anchor={
            "candidate_label": "persistence",
            "candidate_type": "baseline",
            "resolution": "1min",
            "feature_set": "baseline",
            "model_label": "persistence",
            "target_mode": "baseline",
        }
    )

    labels = [str(candidate["candidate_label"]) for candidate in pool]
    assert "minimal_phase_anchor/hgb-frontier-lr010-leaf100/raw+blend" in labels


def test_load_stage5_nowcast_advisory_evidence_reads_transition_support(monkeypatch, tmp_path):
    """Stage-10 should be able to recover broader Stage-5 minute evidence from the latest artifacts."""
    performance_root = tmp_path / "outputs" / "005_performance"
    latest_dir = performance_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "supplemental_surface_advisory.json").write_text(
        json.dumps(
            {
                "candidate_label": "curated_ramp/xgb-balanced/residual+blend",
                "candidate_mae_ratio_to_persistence": 0.964561739,
                "learned_beats_persistence": True,
                "learned_supported_operating_regimes": ["transition_active", "transition_only"],
                "learned_supported_operating_regime_count": 2,
            }
        ),
        encoding="utf-8",
    )
    (latest_dir / "supplemental_surface_segment_evaluation.csv").write_text(
        "\n".join(
            [
                "segment_column,segment_value,candidate_mae_ratio_to_persistence",
                "operating_regime,transition_only,0.939951",
                "operating_regime,transition_active,0.970548",
                "actual_ramp_band,high_ramp,0.878199",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "preferred_output_path", lambda _path: performance_root)

    evidence = module._load_stage5_nowcast_advisory_evidence()

    payload = evidence["curated_ramp/xgb-balanced/residual+blend"]
    assert bool(payload["advisory_surface_supported"]) is True
    assert int(payload["advisory_supported_regime_count"]) == 2
    assert float(payload["advisory_transition_best_ratio_to_persistence"]) == 0.939951
    assert float(payload["advisory_high_ramp_ratio_to_persistence"]) == 0.878199


def test_apply_nowcast_advisory_tie_break_prefers_supported_candidate_within_tolerance(monkeypatch):
    """Broader Stage-5 transition evidence should break only near-tied exact-control nowcast races."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "nowcast_advisory_evidence_enabled", True)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "nowcast_advisory_tie_tolerance", 0.002)
    monkeypatch.setattr(
        module,
        "_load_stage5_nowcast_advisory_evidence",
        lambda: {
            "curated_ramp/xgb-balanced/residual+blend": {
                "advisory_base_candidate_label": "curated_ramp/xgb-balanced/residual+blend",
                "advisory_surface_supported": True,
                "advisory_supported_regime_count": 2,
                "advisory_supported_operating_regimes": ["transition_active", "transition_only"],
                "advisory_surface_candidate_mae_ratio_to_persistence": 0.964562,
                "advisory_transition_only_ratio_to_persistence": 0.939951,
                "advisory_transition_active_ratio_to_persistence": 0.970548,
                "advisory_transition_best_ratio_to_persistence": 0.939951,
                "advisory_high_ramp_ratio_to_persistence": 0.878199,
            }
        },
    )
    benchmark = pd.DataFrame(
        [
            {
                "candidate_label": "full_stable_legacy/hgb-frontier-lr010-leaf100/raw+blend|control_bucket_blend_b5",
                "evaluation_selection_metric_value": 4.698092,
                "evaluation_optimizer_score": 4.698092,
                "evaluation_next_lock_mae": 32.500299,
                "evaluation_peak_interval_miss_rate": 0.125,
                "evaluation_peak_value_mae": 32.136858,
                "evaluation_lock_mae": 47.467989,
            },
            {
                "candidate_label": "curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02",
                "evaluation_selection_metric_value": 4.699189,
                "evaluation_optimizer_score": 4.699189,
                "evaluation_next_lock_mae": 32.483936,
                "evaluation_peak_interval_miss_rate": 0.125,
                "evaluation_peak_value_mae": 31.995877,
                "evaluation_lock_mae": 47.503499,
            },
            {
                "candidate_label": "persistence",
                "evaluation_selection_metric_value": 4.702683,
                "evaluation_optimizer_score": 4.702683,
                "evaluation_next_lock_mae": 32.408185,
                "evaluation_peak_interval_miss_rate": 0.125,
                "evaluation_peak_value_mae": 31.152611,
                "evaluation_lock_mae": 47.456046,
            },
        ]
    )

    ordered, meta = module._apply_nowcast_advisory_tie_break(
        benchmark,
        selection_metric="optimizer_score",
        metric_prefix="evaluation_",
    )

    assert str(ordered.iloc[0]["candidate_label"]) == "curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02"
    assert bool(meta["considered"]) is True
    assert bool(meta["applied"]) is True
    assert str(meta["selected_candidate_label"]) == "curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02"


def test_stage5_candidate_predictions_uses_candidate_specific_blend_config(monkeypatch):
    """Replay Stage-5 blend candidates with their saved candidate-specific blend settings."""
    aligned = pd.DataFrame(
        {
            "y_true": [10.0, 12.0],
            "y_pred": [9.0, 13.0],
            "y_persist": [10.0, 11.0],
        },
        index=[100, 101],
    )
    eval_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-03-01T00:01:00", "2026-03-01T00:02:00"]),
            "avg_load": [10.0, 12.0],
            "lag_1": [10.0, 11.0],
            "feature": [1.0, 2.0],
        },
        index=[100, 101],
    )
    captured: dict[str, float] = {}

    monkeypatch.setattr(module, "_resolve_stage5_feature_set_columns", lambda *_args, **_kwargs: ["feature"])
    monkeypatch.setattr(module, "_fit_stage5_candidate_and_align", lambda **_kwargs: (aligned, 1.0))

    def _fake_apply(*, aligned, blend_config, n_eval_total):
        captured["window"] = float(blend_config.window)
        captured["sharpness"] = float(blend_config.sharpness)
        decisions = pd.DataFrame(
            [
                {"row_index": 100, "blend_pred": 10.0},
                {"row_index": 101, "blend_pred": 12.0},
            ]
        )
        return {"mae": 0.0}, decisions

    monkeypatch.setattr(module, "_apply_stage5_blend_policy", _fake_apply)

    out = module._stage5_candidate_predictions(
        candidate={
            "candidate_label": "minimal_phase_anchor/hgb-frontier-lr010-leaf100/raw+blend",
            "candidate_type": "learned",
            "resolution": "1min",
            "feature_set": "minimal_phase_anchor",
            "model_label": "hgb-frontier-lr010-leaf100",
            "target_mode": "raw+blend",
            "blend_window": 240.0,
            "blend_sharpness": 12.0,
            "blend_min_weight": 0.0,
            "blend_max_weight": 0.6,
        },
        context={
            "train_df": eval_df.copy(),
            "eval_df": eval_df,
            "feature_sets": {"minimal_phase_anchor": ["feature"]},
            "model_catalog": {"hgb-frontier-lr010-leaf100": object()},
            "blend_config": module.Stage5BlendConfig(window=120, sharpness=6.0, min_weight=0.1, max_weight=0.9),
        },
    )

    assert float(captured["window"]) == 240.0
    assert float(captured["sharpness"]) == 12.0
    assert list(out["predicted_load"]) == [10.0, 12.0]


def test_stage5_candidate_predictions_uses_candidate_specific_bucket_blend_config(monkeypatch):
    """Replay Stage-5 bucket-blend candidates with their persisted bucket settings."""
    aligned = pd.DataFrame(
        {
            "y_true": [10.0, 12.0],
            "y_pred": [8.0, 14.0],
            "y_persist": [10.0, 11.0],
        },
        index=[100, 101],
    )
    eval_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-03-01T00:00:00", "2026-03-01T00:05:00"]),
            "avg_load": [10.0, 12.0],
            "lag_1": [10.0, 11.0],
            "feature": [1.0, 2.0],
        },
        index=[100, 101],
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(module, "_resolve_stage5_feature_set_columns", lambda *_args, **_kwargs: ["feature"])
    monkeypatch.setattr(module, "_fit_stage5_candidate_and_align", lambda **_kwargs: (aligned, 1.0))

    def _fake_apply(*, aligned, timestamps, bucket_config, n_eval_total):
        captured["bucket_size_minutes"] = bucket_config.bucket_size_minutes
        captured["cycle_minutes"] = bucket_config.cycle_minutes
        captured["bucket_weights"] = bucket_config.weight_map()
        decisions = pd.DataFrame(
            [
                {"row_index": 100, "blend_pred": 10.0},
                {"row_index": 101, "blend_pred": 12.0},
            ]
        )
        return {"mae": 0.0}, decisions

    monkeypatch.setattr(module, "_apply_stage5_bucket_blend_policy", _fake_apply)

    out = module._stage5_candidate_predictions(
        candidate={
            "candidate_label": "minimal_phase_anchor/hgb-frontier-lr010-leaf100/raw+bucket_blend_b5",
            "candidate_type": "learned",
            "resolution": "1min",
            "feature_set": "minimal_phase_anchor",
            "model_label": "hgb-frontier-lr010-leaf100",
            "target_mode": "raw+bucket_blend_b5",
            "blend_policy_kind": "bucket",
            "blend_bucket_size_minutes": 5.0,
            "blend_bucket_cycle_minutes": 15.0,
            "blend_bucket_weights_json": "{\"0\": 0.4, \"5\": 0.0, \"10\": 0.4}",
        },
        context={
            "train_df": eval_df.copy(),
            "eval_df": eval_df,
            "feature_sets": {"minimal_phase_anchor": ["feature"]},
            "model_catalog": {"hgb-frontier-lr010-leaf100": object()},
            "blend_config": module.Stage5BlendConfig(window=120, sharpness=6.0, min_weight=0.1, max_weight=0.9),
        },
    )

    assert captured["bucket_size_minutes"] == 5
    assert captured["cycle_minutes"] == 15
    assert captured["bucket_weights"] == {0: 0.4, 5: 0.0, 10: 0.4}
    assert list(out["predicted_load"]) == [10.0, 12.0]


def test_stage5_candidate_predictions_uses_bucket_over_sigmoid_config(monkeypatch):
    """Replay Stage-5 bucket candidates that first apply a saved sigmoid blend policy."""
    aligned = pd.DataFrame(
        {
            "y_true": [10.0, 12.0],
            "y_pred": [8.0, 14.0],
            "y_persist": [10.0, 11.0],
        },
        index=[100, 101],
    )
    eval_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-03-01T00:00:00", "2026-03-01T00:05:00"]),
            "avg_load": [10.0, 12.0],
            "lag_1": [10.0, 11.0],
            "feature": [1.0, 2.0],
        },
        index=[100, 101],
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(module, "_resolve_stage5_feature_set_columns", lambda *_args, **_kwargs: ["feature"])
    monkeypatch.setattr(module, "_fit_stage5_candidate_and_align", lambda **_kwargs: (aligned, 1.0))

    def _fake_blend(*, aligned, blend_config, n_eval_total):
        captured["sigmoid_window"] = blend_config.window
        return {"mae": 0.0}, pd.DataFrame(
            [
                {"row_index": 100, "blend_pred": 9.5},
                {"row_index": 101, "blend_pred": 12.5},
            ]
        )

    def _fake_bucket(*, aligned, timestamps, bucket_config, n_eval_total):
        captured["bucket_weights"] = bucket_config.weight_map()
        captured["bucket_input_pred"] = list(aligned["y_pred"].astype(float))
        return {"mae": 0.0}, pd.DataFrame(
            [
                {"row_index": 100, "blend_pred": 10.0},
                {"row_index": 101, "blend_pred": 12.0},
            ]
        )

    monkeypatch.setattr(module, "_apply_stage5_blend_policy", _fake_blend)
    monkeypatch.setattr(module, "_apply_stage5_bucket_blend_policy", _fake_bucket)

    out = module._stage5_candidate_predictions(
        candidate={
            "candidate_label": "minimal_phase_anchor/hgb-frontier-lr010-leaf100/raw+blend_bucket_blend_b5",
            "candidate_type": "learned",
            "resolution": "1min",
            "feature_set": "minimal_phase_anchor",
            "model_label": "hgb-frontier-lr010-leaf100",
            "target_mode": "raw+blend_bucket_blend_b5",
            "blend_policy_kind": "bucket",
            "blend_base_policy_kind": "sigmoid",
            "blend_window": 240.0,
            "blend_sharpness": 12.0,
            "blend_min_weight": 0.0,
            "blend_max_weight": 0.6,
            "blend_bucket_size_minutes": 5.0,
            "blend_bucket_cycle_minutes": 15.0,
            "blend_bucket_weights_json": "{\"0\": 0.4, \"5\": 0.0, \"10\": 0.4}",
        },
        context={
            "train_df": eval_df.copy(),
            "eval_df": eval_df,
            "feature_sets": {"minimal_phase_anchor": ["feature"]},
            "model_catalog": {"hgb-frontier-lr010-leaf100": object()},
            "blend_config": module.Stage5BlendConfig(window=120, sharpness=6.0, min_weight=0.1, max_weight=0.9),
        },
    )

    assert captured["sigmoid_window"] == 240
    assert captured["bucket_weights"] == {0: 0.4, 5: 0.0, 10: 0.4}
    assert captured["bucket_input_pred"] == [9.5, 12.5]
    assert list(out["predicted_load"]) == [10.0, 12.0]


def test_stage5_candidate_predictions_reuses_process_local_cache(monkeypatch):
    """Identical Stage-5 prediction requests should reuse the in-process cache."""
    aligned = pd.DataFrame(
        {
            "y_true": [10.0, 12.0],
            "y_pred": [9.0, 13.0],
            "y_persist": [10.0, 11.0],
        },
        index=[100, 101],
    )
    eval_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-03-01T00:01:00", "2026-03-01T00:02:00"]),
            "avg_load": [10.0, 12.0],
            "lag_1": [10.0, 11.0],
            "feature": [1.0, 2.0],
        },
        index=[100, 101],
    )
    call_counter = {"fit": 0}
    module._STAGE5_PREDICTION_CACHE.clear()

    monkeypatch.setattr(module, "_resolve_stage5_feature_set_columns", lambda *_args, **_kwargs: ["feature"])

    def _fake_fit(**_kwargs):
        call_counter["fit"] += 1
        return aligned, 1.0

    monkeypatch.setattr(module, "_fit_stage5_candidate_and_align", _fake_fit)

    candidate = {
        "candidate_label": "curated/hgb-balanced/raw",
        "candidate_type": "learned",
        "resolution": "1min",
        "feature_set": "curated",
        "model_label": "hgb-balanced",
        "target_mode": "raw",
    }
    context = {
        "context_key": "ctx_a",
        "train_df": eval_df.copy(),
        "eval_df": eval_df,
        "feature_sets": {"curated": ["feature"]},
        "model_catalog": {"hgb-balanced": module.build_model_catalog()["hgb-balanced"]},
        "blend_config": module.Stage5BlendConfig(window=120, sharpness=6.0, min_weight=0.1, max_weight=0.9),
    }

    first = module._stage5_candidate_predictions(candidate=candidate, context=context)
    second = module._stage5_candidate_predictions(candidate=candidate, context=context)

    assert call_counter["fit"] == 1
    assert first.equals(second)


def test_summary_frame_records_day_ahead_relative_improvements_and_nowcast_anchor():
    """Aggregated Stage-10 summary rows should preserve layer gains and nowcast context."""
    by_cycle = pd.DataFrame(
        [
            {
                "day_ahead_minute_path_mae": 10.0,
                "day_ahead_minute_path_mae_pct": 5.0,
                "day_ahead_lock_mae": 8.0,
                "day_ahead_lock_mae_pct": 4.0,
                "day_ahead_next_lock_mae": 7.0,
                "day_ahead_next_lock_mae_pct": 3.5,
                "day_ahead_profile_shape_mae": 7.0,
                "day_ahead_profile_shape_mae_pct": 3.5,
                "day_ahead_energy_mae": 6.0,
                "day_ahead_energy_mae_pct": 3.0,
                "day_ahead_peak_value_mae": 5.0,
                "day_ahead_peak_value_mae_pct": 2.5,
                "day_ahead_peak_interval_hit": 0.0,
                "day_ahead_peak_interval_offset_minutes": 30.0,
                "hourly_minute_path_mae": 9.0,
                "hourly_minute_path_mae_pct": 4.5,
                "hourly_lock_mae": 6.0,
                "hourly_lock_mae_pct": 3.0,
                "hourly_next_lock_mae": 5.0,
                "hourly_next_lock_mae_pct": 2.5,
                "hourly_profile_shape_mae": 6.5,
                "hourly_profile_shape_mae_pct": 3.25,
                "hourly_energy_mae": 5.5,
                "hourly_energy_mae_pct": 2.75,
                "hourly_peak_value_mae": 4.0,
                "hourly_peak_value_mae_pct": 2.0,
                "hourly_peak_interval_hit": 1.0,
                "hourly_peak_interval_offset_minutes": 0.0,
                "phase_minute_path_mae": 8.0,
                "phase_minute_path_mae_pct": 4.0,
                "phase_lock_mae": 5.0,
                "phase_lock_mae_pct": 2.5,
                "phase_next_lock_mae": 4.0,
                "phase_next_lock_mae_pct": 2.0,
                "phase_profile_shape_mae": 5.5,
                "phase_profile_shape_mae_pct": 2.75,
                "phase_energy_mae": 4.5,
                "phase_energy_mae_pct": 2.25,
                "phase_peak_value_mae": 3.0,
                "phase_peak_value_mae_pct": 1.5,
                "phase_peak_interval_hit": 1.0,
                "phase_peak_interval_offset_minutes": 0.0,
                "nowcast_minute_path_mae": 3.0,
                "nowcast_minute_path_mae_pct": 1.5,
                "nowcast_lock_mae": 2.0,
                "nowcast_lock_mae_pct": 1.0,
                "nowcast_next_lock_mae": 1.0,
                "nowcast_next_lock_mae_pct": 0.5,
                "nowcast_profile_shape_mae": 2.5,
                "nowcast_profile_shape_mae_pct": 1.25,
                "nowcast_energy_mae": 1.5,
                "nowcast_energy_mae_pct": 0.75,
                "nowcast_peak_value_mae": 1.0,
                "nowcast_peak_value_mae_pct": 0.5,
                "nowcast_peak_interval_hit": 1.0,
                "nowcast_peak_interval_offset_minutes": 0.0,
            }
        ]
    )
    nowcast_anchor = {
        "candidate_label": "persistence",
        "candidate_type": "baseline",
        "minute_path_mae": 3.0,
        "minute_path_mae_pct": 1.5,
        "control_selection_metric": "lock_mae",
        "control_selection_metric_value": 2.0,
        "control_selection_metric_pct": 1.0,
    }

    summary = module._summary_frame(by_cycle, nowcast_anchor)

    day_ahead = summary.loc[summary["role"].eq("day_ahead")].iloc[0]
    phase = summary.loc[summary["role"].eq("phase")].iloc[0]
    nowcast = summary.loc[summary["role"].eq("nowcast")].iloc[0]
    assert float(day_ahead["lock_mae_gain_vs_day_ahead"]) == 0.0
    assert float(day_ahead["next_lock_mae_gain_vs_day_ahead"]) == 0.0
    assert float(phase["lock_mae_gain_vs_day_ahead"]) == 3.0
    assert float(phase["next_lock_mae_gain_vs_day_ahead"]) == 3.0
    assert float(nowcast["lock_mae_gain_vs_day_ahead"]) == 6.0
    assert float(nowcast["next_lock_mae_gain_vs_day_ahead"]) == 6.0
    assert float(nowcast["peak_interval_hit_rate"]) == 1.0
    assert str(phase["nowcast_anchor_candidate_label"]) == "persistence"
    assert str(nowcast["nowcast_selection_metric"]) == "lock_mae"
    assert float(nowcast["nowcast_selection_metric_value"]) == 2.0


def test_optimizer_delivery_preview_prefers_latest_layer_and_applies_uncertainty():
    """The optimizer preview should expose selected-layer forecasts with calibrated interval bands."""
    interval_timeline = pd.DataFrame(
        [
            {
                "cycle_origin_timestamp": "2026-03-01T00:00:00",
                "interval_start": "2026-03-01T00:00:00",
                "interval_end": "2026-03-01T00:15:00",
                "actual_interval_mean": 105.0,
                "day_ahead_interval_mean": 90.0,
                "hourly_interval_mean": 95.0,
                "phase_interval_mean": 100.0,
                "nowcast_interval_mean": 102.0,
            },
            {
                "cycle_origin_timestamp": "2026-03-01T00:00:00",
                "interval_start": "2026-03-01T00:15:00",
                "interval_end": "2026-03-01T00:30:00",
                "actual_interval_mean": 110.0,
                "day_ahead_interval_mean": 92.0,
                "hourly_interval_mean": 96.0,
                "phase_interval_mean": 101.0,
                "nowcast_interval_mean": 108.0,
            },
        ]
    )
    uncertainty_calibration = pd.DataFrame(
        [
            {
                "layer_role": "nowcast",
                "lead_interval_index": -1,
                "raw_support_n": 16,
                "calibration_sample_n": 16,
                "quantile_source": "layer_global",
                "residual_q025": -8.0,
                "residual_q10": -4.0,
                "residual_q90": 5.0,
                "residual_q975": 9.0,
            },
            {
                "layer_role": "nowcast",
                "lead_interval_index": 0,
                "raw_support_n": 8,
                "calibration_sample_n": 8,
                "quantile_source": "lead_interval",
                "residual_q025": -6.0,
                "residual_q10": -3.0,
                "residual_q90": 4.0,
                "residual_q975": 7.0,
            },
            {
                "layer_role": "nowcast",
                "lead_interval_index": 1,
                "raw_support_n": 8,
                "calibration_sample_n": 8,
                "quantile_source": "lead_interval",
                "residual_q025": -5.0,
                "residual_q10": -2.0,
                "residual_q90": 3.0,
                "residual_q975": 6.0,
            },
        ]
    )
    policy = {
        "day_ahead": {"candidate_label": "day_ahead_model"},
        "hourly": {"candidate_label": "hourly_model"},
        "phase": {"candidate_label": "phase_model"},
        "nowcast_anchor": {"candidate_label": "nowcast_model"},
    }

    preview = module._build_optimizer_delivery_preview(
        interval_timeline=interval_timeline,
        policy=policy,
        uncertainty_calibration=uncertainty_calibration,
        lock_interval_minutes=15,
        run_id="run_123",
        config_hash="sha256:test",
    )

    first_row = preview.iloc[0]
    assert str(first_row["producer_stage"]) == "010_forecast_control"
    assert str(first_row["contract_version"]) == module.OPTIMIZER_CONTRACT_VERSION
    assert str(first_row["run_id"]) == "run_123"
    assert str(first_row["config_hash"]) == "sha256:test"
    assert str(first_row["selected_layer_role"]) == "nowcast"
    assert str(first_row["selected_candidate_label"]) == "nowcast_model"
    assert str(first_row["as_of_timestamp"]) == "2026-03-01 00:00:00"
    assert float(first_row["expected_layer_cadence_minutes"]) == 1.0
    assert float(first_row["forecast_age_minutes"]) == 0.0
    assert float(first_row["stale_threshold_minutes"]) == 5.0
    assert bool(first_row["is_stale_forecast"]) is False
    assert float(first_row["forecast_value"]) == 102.0
    assert float(first_row["forecast_lower_80"]) == 99.0
    assert float(first_row["forecast_upper_80"]) == 106.0
    assert float(first_row["confidence_score"]) > 0.75
    assert str(first_row["confidence_tier"]) == "high"
    assert bool(first_row["is_next_lock_interval"]) is True
    assert bool(first_row["within_95_band"]) is True


def test_interval_timeline_for_cycle_adds_dynamic_operating_context(monkeypatch):
    """Stage-10 interval rows should carry transition/activity/ramp context for dynamic overlay routing."""
    monkeypatch.setattr(
        module,
        "_load_optimizer_delivery_context_by_timestamp",
        lambda: pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2026-03-01 00:00:00"),
                    "workday_transition": 1.0,
                    "profile_active_flag": 1.0,
                    "ramp_flag": 1.0,
                },
                {
                    "timestamp": pd.Timestamp("2026-03-01 00:01:00"),
                    "workday_transition": 1.0,
                    "profile_active_flag": 1.0,
                    "ramp_flag": 0.0,
                },
                {
                    "timestamp": pd.Timestamp("2026-03-01 00:02:00"),
                    "workday_transition": 0.0,
                    "profile_active_flag": 1.0,
                    "ramp_flag": 0.0,
                },
            ]
        ),
    )
    minute_frame = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-01 00:00:00"),
                "actual_load": 100.0,
                "day_ahead_pred": 90.0,
                "hourly_pred": 95.0,
                "phase_pred": 98.0,
                "nowcast_pred": 99.0,
            },
            {
                "timestamp": pd.Timestamp("2026-03-01 00:01:00"),
                "actual_load": 102.0,
                "day_ahead_pred": 91.0,
                "hourly_pred": 96.0,
                "phase_pred": 99.0,
                "nowcast_pred": 100.0,
            },
            {
                "timestamp": pd.Timestamp("2026-03-01 00:02:00"),
                "actual_load": 104.0,
                "day_ahead_pred": 92.0,
                "hourly_pred": 97.0,
                "phase_pred": 100.0,
                "nowcast_pred": 101.0,
            },
        ]
    )

    interval_timeline = module._interval_timeline_for_cycle(
        cycle_origin_timestamp=pd.Timestamp("2026-03-01 00:00:00"),
        minute_frame=minute_frame,
        lock_interval_minutes=15,
    )

    row = interval_timeline.iloc[0]
    assert bool(row["workday_transition_active"]) is True
    assert float(row["profile_active_fraction"]) == pytest.approx(1.0)
    assert float(row["high_ramp_fraction"]) == pytest.approx(1.0 / 3.0)
    assert str(row["actual_ramp_band"]) == "high_ramp"
    assert str(row["operating_regime"]) == "transition_active"


def test_optimizer_delivery_preview_dynamically_demotes_background_nowcast():
    """The learned minute overlay should fall back on background intervals where broader evidence says it is weak."""
    original_enforce = module.MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_enforce"]
    module.MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_enforce"] = True
    try:
        interval_timeline = pd.DataFrame(
            [
                {
                    "cycle_origin_timestamp": "2026-03-01T00:00:00",
                    "interval_start": "2026-03-01T00:00:00",
                    "interval_end": "2026-03-01T00:15:00",
                    "actual_interval_mean": 150.0,
                    "day_ahead_interval_mean": 120.0,
                    "hourly_interval_mean": 130.0,
                    "phase_interval_mean": 135.0,
                    "nowcast_interval_mean": 145.0,
                    "operating_regime": "transition_only",
                    "actual_ramp_band": "stable_ramp",
                    "high_ramp_fraction": 0.0,
                },
                {
                    "cycle_origin_timestamp": "2026-03-01T00:00:00",
                    "interval_start": "2026-03-01T00:15:00",
                    "interval_end": "2026-03-01T00:30:00",
                    "actual_interval_mean": 100.0,
                    "day_ahead_interval_mean": 90.0,
                    "hourly_interval_mean": 94.0,
                    "phase_interval_mean": 95.0,
                    "nowcast_interval_mean": 96.0,
                    "operating_regime": "none_inactive",
                    "actual_ramp_band": "stable_ramp",
                    "high_ramp_fraction": 0.0,
                },
            ]
        )
        policy = {
            "day_ahead": {"candidate_label": "day_ahead_model"},
            "hourly": {"candidate_label": "hourly_model"},
            "phase": {"candidate_label": "phase_model"},
            "nowcast_anchor": {
                "candidate_label": "curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02",
                "advisory_surface_supported": True,
                "advisory_supported_operating_regimes": ["transition_only", "transition_active"],
                "advisory_transition_best_ratio_to_persistence": 0.939951,
                "advisory_high_ramp_ratio_to_persistence": 0.878199,
            },
        }

        preview = module._build_optimizer_delivery_preview(
            interval_timeline=interval_timeline,
            policy=policy,
            uncertainty_calibration=pd.DataFrame(),
            lock_interval_minutes=15,
            run_id="run_dynamic_demote",
            config_hash="sha256:dynamic-demote",
        )

        background_row = preview.iloc[1]
        assert bool(background_row["nowcast_dynamic_overlay_enforced"]) is True
        assert bool(background_row["nowcast_dynamic_overlay_eligible"]) is False
        assert str(background_row["nowcast_dynamic_overlay_reason"]) == "background_interval"
        assert str(background_row["selected_layer_role"]) == "phase"
        assert str(background_row["selected_candidate_label"]) == "phase_model"
        assert str(background_row["fallback_trigger"]) == "dynamic_gate"
        assert str(background_row["fallback_reason"]) == "nowcast_dynamic_gate_background_interval"
    finally:
        module.MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_enforce"] = original_enforce


def test_optimizer_delivery_preview_keeps_nowcast_on_next_lock_even_in_background_regime():
    """The dynamic minute controller should preserve nowcast on next-lock intervals where Stage-10 value is strongest."""
    original_enforce = module.MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_enforce"]
    module.MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_enforce"] = True
    try:
        interval_timeline = pd.DataFrame(
            [
                {
                    "cycle_origin_timestamp": "2026-03-01T00:00:00",
                    "interval_start": "2026-03-01T00:00:00",
                    "interval_end": "2026-03-01T00:15:00",
                    "actual_interval_mean": 103.0,
                    "day_ahead_interval_mean": 90.0,
                    "hourly_interval_mean": 95.0,
                    "phase_interval_mean": 100.0,
                    "nowcast_interval_mean": 102.0,
                    "operating_regime": "none_inactive",
                    "actual_ramp_band": "stable_ramp",
                    "high_ramp_fraction": 0.0,
                }
            ]
        )
        policy = {
            "day_ahead": {"candidate_label": "day_ahead_model"},
            "hourly": {"candidate_label": "hourly_model"},
            "phase": {"candidate_label": "phase_model"},
            "nowcast_anchor": {
                "candidate_label": "curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02",
                "advisory_surface_supported": True,
                "advisory_supported_operating_regimes": ["transition_only", "transition_active"],
                "advisory_transition_best_ratio_to_persistence": 0.939951,
                "advisory_high_ramp_ratio_to_persistence": 0.878199,
            },
        }

        preview = module._build_optimizer_delivery_preview(
            interval_timeline=interval_timeline,
            policy=policy,
            uncertainty_calibration=pd.DataFrame(),
            lock_interval_minutes=15,
            run_id="run_dynamic_keep",
            config_hash="sha256:dynamic-keep",
        )

        row = preview.iloc[0]
        assert bool(row["is_next_lock_interval"]) is True
        assert bool(row["nowcast_dynamic_overlay_enforced"]) is True
        assert bool(row["nowcast_dynamic_overlay_eligible"]) is True
        assert str(row["nowcast_dynamic_overlay_reason"]) == "next_lock_interval"
        assert str(row["selected_layer_role"]) == "nowcast"
    finally:
        module.MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_enforce"] = original_enforce


def test_build_optimizer_delivery_serving_preview_keeps_dynamic_overlay_fields():
    """Serving-shaped previews should retain the dynamic minute-controller audit fields."""
    preview = pd.DataFrame(
        [
            {
                "producer_stage": "010_forecast_control",
                "contract_version": "1.2",
                "run_id": "run_dynamic_fields",
                "config_hash": "sha256:dynamic-fields",
                "cycle_origin_timestamp": "2026-03-01T00:00:00",
                "as_of_timestamp": "2026-03-01T00:00:00",
                "interval_start": "2026-03-01T00:00:00",
                "interval_end": "2026-03-01T00:15:00",
                "lead_interval_index": 0,
                "horizon_minutes": 1,
                "is_next_lock_interval": True,
                "requested_is_predicted_peak_interval": True,
                "operating_regime": "transition_only",
                "actual_ramp_band": "high_ramp",
                "high_ramp_fraction": 0.5,
                "effective_forecast_as_of": "2026-03-01T00:00:00",
                "requested_layer_role": "nowcast",
                "requested_candidate_label": "nowcast_model",
                "nowcast_dynamic_overlay_enabled": True,
                "nowcast_dynamic_overlay_enforced": False,
                "nowcast_dynamic_overlay_eligible": True,
                "nowcast_dynamic_overlay_reason": "next_lock_interval",
                "selected_layer_role": "nowcast",
                "selected_layer": "Nowcast",
                "selected_candidate_label": "nowcast_model",
                "expected_layer_cadence_minutes": 1,
                "forecast_age_minutes": 0.0,
                "stale_threshold_minutes": 2,
                "is_stale_forecast": False,
                "fallback_applied": False,
                "fallback_from_layer_role": "",
                "fallback_to_layer_role": "",
                "fallback_trigger": "none",
                "fallback_reason": "full_stack_available",
                "resolution_path": "nowcast:ready",
                "forecast_value": 123.0,
                "forecast_lower_80": 118.0,
                "forecast_upper_80": 128.0,
                "forecast_lower_95": 115.0,
                "forecast_upper_95": 131.0,
                "uncertainty_band_width_80": 10.0,
                "uncertainty_band_width_95": 16.0,
                "uncertainty_band_width_95_pct": 13.0,
                "quantile_source": "lead_interval",
                "calibration_sample_n": 16,
                "raw_support_n": 16,
                "confidence_score": 0.8,
                "confidence_tier": "high",
            }
        ]
    )

    serving = module._build_optimizer_delivery_serving_preview(preview)

    assert "operating_regime" in serving.columns
    assert "actual_ramp_band" in serving.columns
    assert "high_ramp_fraction" in serving.columns
    assert "nowcast_dynamic_overlay_enabled" in serving.columns
    assert "nowcast_dynamic_overlay_enforced" in serving.columns
    assert "nowcast_dynamic_overlay_eligible" in serving.columns
    assert "nowcast_dynamic_overlay_reason" in serving.columns


def test_build_optimizer_dynamic_overlay_shadow_summary_recommends_shadow_when_enforcement_hurts():
    """The persisted shadow summary should reject enforcement when dynamic demotion degrades the replay surface."""
    interval_timeline = pd.DataFrame(
        [
            {
                "cycle_origin_timestamp": "2026-03-01T00:00:00",
                "interval_start": "2026-03-01T00:00:00",
                "interval_end": "2026-03-01T00:15:00",
                "actual_interval_mean": 150.0,
                "day_ahead_interval_mean": 120.0,
                "hourly_interval_mean": 130.0,
                "phase_interval_mean": 135.0,
                "nowcast_interval_mean": 145.0,
                "operating_regime": "transition_only",
                "actual_ramp_band": "stable_ramp",
                "high_ramp_fraction": 0.0,
            },
            {
                "cycle_origin_timestamp": "2026-03-01T00:00:00",
                "interval_start": "2026-03-01T00:15:00",
                "interval_end": "2026-03-01T00:30:00",
                "actual_interval_mean": 100.0,
                "day_ahead_interval_mean": 90.0,
                "hourly_interval_mean": 94.0,
                "phase_interval_mean": 80.0,
                "nowcast_interval_mean": 99.0,
                "operating_regime": "none_inactive",
                "actual_ramp_band": "stable_ramp",
                "high_ramp_fraction": 0.0,
            },
        ]
    )
    policy = {
        "day_ahead": {"candidate_label": "day_ahead_model"},
        "hourly": {"candidate_label": "hourly_model"},
        "phase": {"candidate_label": "phase_model"},
        "nowcast_anchor": {
            "candidate_label": "curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02",
            "advisory_surface_supported": True,
            "advisory_supported_operating_regimes": ["transition_only", "transition_active"],
            "advisory_transition_best_ratio_to_persistence": 0.939951,
            "advisory_high_ramp_ratio_to_persistence": 0.878199,
        },
    }

    preview = module._build_optimizer_delivery_preview(
        interval_timeline=interval_timeline,
        policy=policy,
        uncertainty_calibration=pd.DataFrame(),
        lock_interval_minutes=15,
        run_id="run_dynamic_shadow_summary",
        config_hash="sha256:dynamic-shadow-summary",
    )

    summary = module._build_optimizer_dynamic_overlay_shadow_summary(preview)

    assert summary["enabled"] is True
    assert summary["recommendation"] == "keep_shadow_mode"
    assert summary["shadow_mode"]["selected_layer_counts"]["nowcast"] == 2
    assert summary["enforced_counterfactual"]["selected_layer_counts"]["phase"] == 1
    assert summary["delta_enforced_minus_shadow"]["mean_selected_abs_error"] > 0.0


def test_apply_optimizer_dynamic_soft_overlay_candidate_keeps_strategic_rows_full_nowcast():
    """Soft overlay shadow logic should keep strategic rows at full nowcast weight."""
    preview = pd.DataFrame(
        [
            {
                "cycle_origin_timestamp": "2026-03-01T00:00:00",
                "interval_start": "2026-03-01T00:00:00",
                "actual_interval_mean": 150.0,
                "phase_interval_mean": 130.0,
                "hourly_interval_mean": 125.0,
                "nowcast_interval_mean": 148.0,
                "forecast_value": 148.0,
                "selected_abs_error": 2.0,
                "requested_layer_role": "nowcast",
                "selected_layer_role": "nowcast",
                "is_next_lock_interval": True,
                "requested_is_predicted_peak_interval": False,
                "nowcast_dynamic_overlay_enabled": True,
                "nowcast_dynamic_overlay_eligible": True,
                "nowcast_dynamic_overlay_reason": "next_lock_interval",
            },
            {
                "cycle_origin_timestamp": "2026-03-01T00:00:00",
                "interval_start": "2026-03-01T00:15:00",
                "actual_interval_mean": 100.0,
                "phase_interval_mean": 80.0,
                "hourly_interval_mean": 82.0,
                "nowcast_interval_mean": 92.0,
                "forecast_value": 92.0,
                "selected_abs_error": 8.0,
                "requested_layer_role": "nowcast",
                "selected_layer_role": "nowcast",
                "is_next_lock_interval": False,
                "requested_is_predicted_peak_interval": False,
                "nowcast_dynamic_overlay_enabled": True,
                "nowcast_dynamic_overlay_eligible": True,
                "nowcast_dynamic_overlay_reason": "supported_operating_regime",
            },
            {
                "cycle_origin_timestamp": "2026-03-01T00:00:00",
                "interval_start": "2026-03-01T00:30:00",
                "actual_interval_mean": 100.0,
                "phase_interval_mean": 80.0,
                "hourly_interval_mean": 82.0,
                "nowcast_interval_mean": 110.0,
                "forecast_value": 110.0,
                "selected_abs_error": 10.0,
                "requested_layer_role": "nowcast",
                "selected_layer_role": "nowcast",
                "is_next_lock_interval": False,
                "requested_is_predicted_peak_interval": False,
                "nowcast_dynamic_overlay_enabled": True,
                "nowcast_dynamic_overlay_eligible": False,
                "nowcast_dynamic_overlay_reason": "background_interval",
            },
        ]
    )

    soft = module._apply_optimizer_dynamic_soft_overlay_candidate(
        preview,
        supported_weight=0.8,
        background_weight=0.2,
    )

    assert float(soft.loc[0, "nowcast_dynamic_soft_weight"]) == pytest.approx(1.0)
    assert float(soft.loc[1, "nowcast_dynamic_soft_weight"]) == pytest.approx(0.8)
    assert float(soft.loc[2, "nowcast_dynamic_soft_weight"]) == pytest.approx(0.2)
    assert str(soft.loc[0, "nowcast_dynamic_soft_bucket"]) == "strategic"
    assert str(soft.loc[1, "nowcast_dynamic_soft_bucket"]) == "supported"
    assert str(soft.loc[2, "nowcast_dynamic_soft_bucket"]) == "background"
    assert float(soft.loc[0, "forecast_value"]) == pytest.approx(148.0)


def test_build_optimizer_dynamic_overlay_soft_summary_recommends_positive_candidate():
    """Soft overlay shadow search should recommend a blended background policy when it improves replay."""
    original_supported = list(module.MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_supported_weights"])
    original_background = list(module.MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_background_weights"])
    try:
        module.MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_supported_weights"] = [1.0]
        module.MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_background_weights"] = [0.0, 0.65, 1.0]
        preview = pd.DataFrame(
            [
                {
                    "cycle_origin_timestamp": "2026-03-01T00:00:00",
                    "interval_start": "2026-03-01T00:00:00",
                    "actual_interval_mean": 120.0,
                    "phase_interval_mean": 90.0,
                    "hourly_interval_mean": 92.0,
                    "nowcast_interval_mean": 118.0,
                    "forecast_value": 118.0,
                    "selected_abs_error": 2.0,
                    "requested_layer_role": "nowcast",
                    "selected_layer_role": "nowcast",
                    "is_next_lock_interval": True,
                    "requested_is_predicted_peak_interval": True,
                    "is_actual_peak_interval": True,
                    "is_predicted_peak_interval": True,
                    "nowcast_dynamic_overlay_enabled": True,
                    "nowcast_dynamic_overlay_eligible": True,
                    "nowcast_dynamic_overlay_reason": "next_lock_interval",
                },
                {
                    "cycle_origin_timestamp": "2026-03-01T00:00:00",
                    "interval_start": "2026-03-01T00:15:00",
                    "actual_interval_mean": 100.0,
                    "phase_interval_mean": 80.0,
                    "hourly_interval_mean": 82.0,
                    "nowcast_interval_mean": 110.0,
                    "forecast_value": 110.0,
                    "selected_abs_error": 10.0,
                    "requested_layer_role": "nowcast",
                    "selected_layer_role": "nowcast",
                    "is_next_lock_interval": False,
                    "requested_is_predicted_peak_interval": False,
                    "is_actual_peak_interval": False,
                    "is_predicted_peak_interval": False,
                    "nowcast_dynamic_overlay_enabled": True,
                    "nowcast_dynamic_overlay_eligible": False,
                    "nowcast_dynamic_overlay_reason": "background_interval",
                },
            ]
        )

        candidates = module._evaluate_optimizer_dynamic_soft_overlay_candidates(preview)
        summary = module._build_optimizer_dynamic_overlay_soft_summary(preview, candidates)

        assert not candidates.empty
        assert summary["recommendation"] == "shadow_soft_overlay_candidate_positive"
        assert summary["best_improving_candidate"]["background_weight"] == pytest.approx(0.65)
        assert summary["best_improving_candidate"]["delta_mean_selected_abs_error_vs_shadow"] < 0.0
        assert summary["best_improving_candidate"]["delta_next_lock_mae_vs_shadow"] == pytest.approx(0.0)
    finally:
        module.MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_supported_weights"] = original_supported
        module.MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_background_weights"] = original_background


def test_optimizer_delivery_preview_recovers_advisory_support_from_stage5_when_policy_metadata_is_sparse(
    monkeypatch,
):
    """The dynamic minute controller should recover advisory support from Stage-5 artifacts when needed."""
    monkeypatch.setattr(
        module,
        "_load_stage5_nowcast_advisory_evidence",
        lambda: {
            "curated_ramp/xgb-balanced/residual+blend": {
                "advisory_surface_supported": True,
                "advisory_supported_operating_regimes": ["transition_only", "transition_active"],
                "advisory_supported_regime_count": 2,
                "advisory_transition_best_ratio_to_persistence": 0.939951,
                "advisory_high_ramp_ratio_to_persistence": 0.878199,
            }
        },
    )
    interval_timeline = pd.DataFrame(
        [
            {
                "cycle_origin_timestamp": "2026-03-01T00:00:00",
                "interval_start": "2026-03-01T00:00:00",
                "interval_end": "2026-03-01T00:15:00",
                "actual_interval_mean": 150.0,
                "day_ahead_interval_mean": 120.0,
                "hourly_interval_mean": 130.0,
                "phase_interval_mean": 135.0,
                "nowcast_interval_mean": 145.0,
                "operating_regime": "transition_only",
                "actual_ramp_band": "stable_ramp",
                "high_ramp_fraction": 0.0,
            }
        ]
    )
    policy = {
        "day_ahead": {"candidate_label": "day_ahead_model"},
        "hourly": {"candidate_label": "hourly_model"},
        "phase": {"candidate_label": "phase_model"},
        "nowcast_anchor": {
            "candidate_label": "curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02",
            "blend_base_candidate_label": "curated_ramp/xgb-balanced/residual+blend",
        },
    }

    preview = module._build_optimizer_delivery_preview(
        interval_timeline=interval_timeline,
        policy=policy,
        uncertainty_calibration=pd.DataFrame(),
        lock_interval_minutes=15,
        run_id="run_dynamic_advisory_fallback",
        config_hash="sha256:dynamic-advisory-fallback",
    )

    row = preview.iloc[0]
    assert bool(row["nowcast_dynamic_overlay_enabled"]) is True
    assert bool(row["nowcast_dynamic_overlay_eligible"]) is True
    assert str(row["nowcast_dynamic_overlay_reason"]) in {"transition_only", "supported_operating_regime", "next_lock_interval"}


def test_optimizer_delivery_preview_uses_predicted_peak_specific_quantiles():
    """Predicted-peak rows should prefer the peak-specific uncertainty band when available."""
    interval_timeline = pd.DataFrame(
        [
            {
                "cycle_origin_timestamp": "2026-03-01T00:00:00",
                "interval_start": "2026-03-01T00:00:00",
                "interval_end": "2026-03-01T00:15:00",
                "actual_interval_mean": 100.0,
                "day_ahead_interval_mean": 80.0,
                "hourly_interval_mean": 85.0,
                "phase_interval_mean": 90.0,
                "nowcast_interval_mean": 95.0,
            },
            {
                "cycle_origin_timestamp": "2026-03-01T00:00:00",
                "interval_start": "2026-03-01T00:15:00",
                "interval_end": "2026-03-01T00:30:00",
                "actual_interval_mean": 205.0,
                "day_ahead_interval_mean": 180.0,
                "hourly_interval_mean": 185.0,
                "phase_interval_mean": 190.0,
                "nowcast_interval_mean": 200.0,
            },
        ]
    )
    uncertainty_calibration = pd.DataFrame(
        [
            {
                "layer_role": "nowcast",
                "lead_interval_index": -1,
                "raw_support_n": 16,
                "calibration_sample_n": 16,
                "peak_context": "all",
                "quantile_source": "layer_global",
                "residual_q025": -12.0,
                "residual_q10": -10.0,
                "residual_q90": 10.0,
                "residual_q975": 12.0,
            },
            {
                "layer_role": "nowcast",
                "lead_interval_index": -1,
                "raw_support_n": 8,
                "calibration_sample_n": 8,
                "peak_context": "predicted_peak",
                "quantile_source": "predicted_peak_global",
                "residual_q025": -4.0,
                "residual_q10": -2.0,
                "residual_q90": 2.0,
                "residual_q975": 4.0,
            },
        ]
    )
    policy = {
        "day_ahead": {"candidate_label": "day_ahead_model"},
        "hourly": {"candidate_label": "hourly_model"},
        "phase": {"candidate_label": "phase_model"},
        "nowcast_anchor": {"candidate_label": "nowcast_model"},
    }

    preview = module._build_optimizer_delivery_preview(
        interval_timeline=interval_timeline,
        policy=policy,
        uncertainty_calibration=uncertainty_calibration,
        lock_interval_minutes=15,
        run_id="run_peak",
        config_hash="sha256:peak",
    )

    peak_row = preview.iloc[1]
    assert bool(peak_row["is_predicted_peak_interval"]) is True
    assert str(peak_row["quantile_source"]) == "predicted_peak_global"
    assert float(peak_row["forecast_lower_80"]) == 198.0
    assert float(peak_row["forecast_upper_80"]) == 202.0


def test_optimizer_delivery_preview_live_falls_back_to_phase_when_nowcast_is_stale():
    """Live delivery should resolve to the next older layer when the freshest layer is stale."""
    interval_timeline = pd.DataFrame(
        [
            {
                "cycle_origin_timestamp": "2026-03-01T00:00:00",
                "interval_start": "2026-03-01T00:00:00",
                "interval_end": "2026-03-01T00:15:00",
                "actual_interval_mean": 103.0,
                "day_ahead_interval_mean": 90.0,
                "hourly_interval_mean": 95.0,
                "phase_interval_mean": 100.0,
                "nowcast_interval_mean": 102.0,
            }
        ]
    )
    uncertainty_calibration = pd.DataFrame(
        [
            {
                "layer_role": "phase",
                "lead_interval_index": 0,
                "peak_context": "all",
                "raw_support_n": 8,
                "calibration_sample_n": 8,
                "quantile_source": "lead_interval",
                "residual_q025": -4.0,
                "residual_q10": -2.0,
                "residual_q90": 3.0,
                "residual_q975": 5.0,
            },
            {
                "layer_role": "phase",
                "lead_interval_index": -1,
                "peak_context": "all",
                "raw_support_n": 16,
                "calibration_sample_n": 16,
                "quantile_source": "layer_global",
                "residual_q025": -6.0,
                "residual_q10": -3.0,
                "residual_q90": 4.0,
                "residual_q975": 7.0,
            },
            {
                "layer_role": "nowcast",
                "lead_interval_index": 0,
                "peak_context": "all",
                "raw_support_n": 8,
                "calibration_sample_n": 8,
                "quantile_source": "lead_interval",
                "residual_q025": -2.0,
                "residual_q10": -1.0,
                "residual_q90": 1.0,
                "residual_q975": 2.0,
            },
            {
                "layer_role": "nowcast",
                "lead_interval_index": -1,
                "peak_context": "all",
                "raw_support_n": 16,
                "calibration_sample_n": 16,
                "quantile_source": "layer_global",
                "residual_q025": -3.0,
                "residual_q10": -2.0,
                "residual_q90": 2.0,
                "residual_q975": 3.0,
            },
        ]
    )
    policy = {
        "day_ahead": {"candidate_label": "day_ahead_model"},
        "hourly": {"candidate_label": "hourly_model"},
        "phase": {"candidate_label": "phase_model"},
        "nowcast_anchor": {"candidate_label": "nowcast_model"},
    }

    preview = module._build_optimizer_delivery_preview(
        interval_timeline=interval_timeline,
        policy=policy,
        uncertainty_calibration=uncertainty_calibration,
        lock_interval_minutes=15,
        run_id="run_live",
        config_hash="sha256:live",
        delivery_as_of_timestamp="2026-03-01T00:10:00",
    )

    row = preview.iloc[0]
    assert str(row["requested_layer_role"]) == "nowcast"
    assert str(row["selected_layer_role"]) == "phase"
    assert str(row["selected_candidate_label"]) == "phase_model"
    assert bool(row["fallback_applied"]) is True
    assert str(row["fallback_trigger"]) == "stale"
    assert str(row["fallback_reason"]) == "nowcast_stale"
    assert float(row["forecast_value"]) == 100.0
    assert str(row["quantile_source"]) == "lead_interval"
    assert bool(row["is_stale_forecast"]) is False


def test_optimizer_delivery_uncertainty_calibration_adds_context_specific_rows(monkeypatch):
    """Calibration should emit next-lock and predicted-peak lead-specific buckets when support exists."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "optimizer_delivery_min_lead_specific_samples", 2)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "optimizer_delivery_next_lock_min_samples", 2)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "optimizer_delivery_next_lock_scaled_enabled", True)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "optimizer_delivery_next_lock_scale_floor_quantile", 0.25)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "optimizer_delivery_next_lock_scale_floor_min_load", 50.0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "optimizer_delivery_predicted_peak_min_samples", 2)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "optimizer_delivery_predicted_peak_lead_min_samples", 2)
    calibration_interval_timeline = pd.DataFrame(
        [
            {
                "cycle_origin_timestamp": "2026-03-01T00:00:00",
                "interval_start": "2026-03-01T00:00:00",
                "interval_end": "2026-03-01T00:15:00",
                "actual_interval_mean": 100.0,
                "nowcast_interval_mean": 98.0,
            },
            {
                "cycle_origin_timestamp": "2026-03-01T00:00:00",
                "interval_start": "2026-03-01T00:15:00",
                "interval_end": "2026-03-01T00:30:00",
                "actual_interval_mean": 140.0,
                "nowcast_interval_mean": 135.0,
            },
            {
                "cycle_origin_timestamp": "2026-03-01T06:00:00",
                "interval_start": "2026-03-01T06:00:00",
                "interval_end": "2026-03-01T06:15:00",
                "actual_interval_mean": 101.0,
                "nowcast_interval_mean": 99.0,
            },
            {
                "cycle_origin_timestamp": "2026-03-01T06:00:00",
                "interval_start": "2026-03-01T06:15:00",
                "interval_end": "2026-03-01T06:30:00",
                "actual_interval_mean": 141.0,
                "nowcast_interval_mean": 136.0,
            },
        ]
    )

    calibration = module._build_optimizer_delivery_uncertainty_calibration(
        calibration_interval_timeline=calibration_interval_timeline,
        lock_interval_minutes=15,
    )

    sources = set(calibration["quantile_source"].astype("string"))
    assert "next_lock_global" in sources
    assert "next_lock_scaled_global" in sources
    assert "predicted_peak_lead_interval" in sources


def test_optimizer_delivery_preview_scales_next_lock_uncertainty_by_forecast_size():
    """Scaled next-lock calibration should produce narrower absolute bands on smaller forecasts."""
    interval_timeline = pd.DataFrame(
        [
            {
                "cycle_origin_timestamp": "2026-03-01T00:00:00",
                "interval_start": "2026-03-01T00:00:00",
                "interval_end": "2026-03-01T00:15:00",
                "actual_interval_mean": 310.0,
                "day_ahead_interval_mean": 290.0,
                "hourly_interval_mean": 295.0,
                "phase_interval_mean": 298.0,
                "nowcast_interval_mean": 300.0,
            },
            {
                "cycle_origin_timestamp": "2026-03-01T06:00:00",
                "interval_start": "2026-03-01T06:00:00",
                "interval_end": "2026-03-01T06:15:00",
                "actual_interval_mean": 2620.0,
                "day_ahead_interval_mean": 2550.0,
                "hourly_interval_mean": 2575.0,
                "phase_interval_mean": 2590.0,
                "nowcast_interval_mean": 2600.0,
            },
        ]
    )
    uncertainty_calibration = pd.DataFrame(
        [
            {
                "layer_role": "nowcast",
                "lead_interval_index": -1,
                "lead_interval_start_minutes": 0,
                "lead_interval_end_minutes": 15,
                "peak_context": "next_lock",
                "raw_support_n": 12,
                "calibration_sample_n": 12,
                "quantile_source": "next_lock_scaled_global",
                "residual_q025": -0.10,
                "residual_q10": -0.05,
                "residual_q90": 0.10,
                "residual_q975": 0.20,
                "residual_scale_mode": "absolute_prediction_floor",
                "scale_floor_value": 250.0,
            },
            {
                "layer_role": "nowcast",
                "lead_interval_index": 0,
                "lead_interval_start_minutes": 0,
                "lead_interval_end_minutes": 15,
                "peak_context": "all",
                "raw_support_n": 12,
                "calibration_sample_n": 12,
                "quantile_source": "lead_interval",
                "residual_q025": -500.0,
                "residual_q10": -250.0,
                "residual_q90": 500.0,
                "residual_q975": 1000.0,
            },
        ]
    )
    policy = {
        "day_ahead": {"candidate_label": "day_ahead_model"},
        "hourly": {"candidate_label": "hourly_model"},
        "phase": {"candidate_label": "phase_model"},
        "nowcast_anchor": {"candidate_label": "nowcast_model"},
    }

    preview = module._build_optimizer_delivery_preview(
        interval_timeline=interval_timeline,
        policy=policy,
        uncertainty_calibration=uncertainty_calibration,
        lock_interval_minutes=15,
        run_id="run_scaled",
        config_hash="sha256:scaled",
    )

    low_row = preview.iloc[0]
    high_row = preview.iloc[1]
    assert str(low_row["quantile_source"]) == "next_lock_scaled_global"
    assert str(low_row["residual_scale_mode"]) == "absolute_prediction_floor"
    assert float(low_row["forecast_lower_95"]) == 270.0
    assert float(low_row["forecast_upper_95"]) == 360.0
    assert float(high_row["forecast_lower_95"]) == 2340.0
    assert float(high_row["forecast_upper_95"]) == 3120.0
    assert float(low_row["uncertainty_band_width_95"]) < float(high_row["uncertainty_band_width_95"])


def test_apply_optimizer_delivery_staleness_flags_old_rows_and_recomputes_confidence():
    """Wall-clock staleness should downgrade the trust surface for cached delivery rows."""
    preview = pd.DataFrame(
        [
            {
                "as_of_timestamp": "2026-03-01T00:00:00",
                "effective_forecast_as_of": "2026-03-01T00:00:00",
                "selected_layer_role": "nowcast",
                "forecast_age_minutes": 0.0,
                "stale_threshold_minutes": 5,
                "is_stale_forecast": False,
                "uncertainty_band_width_95_pct": 20.0,
                "calibration_sample_n": 8,
                "quantile_source": "lead_interval",
                "confidence_score": 0.8,
                "confidence_tier": "high",
            }
        ]
    )

    stale_preview = module._apply_optimizer_delivery_staleness(
        preview,
        as_of_timestamp="2026-03-01T00:20:00",
    )

    row = stale_preview.iloc[0]
    assert str(row["as_of_timestamp"]) == "2026-03-01 00:20:00"
    assert float(row["forecast_age_minutes"]) == 20.0
    assert bool(row["is_stale_forecast"]) is True
    assert float(row["confidence_score"]) < 0.8
    assert str(row["confidence_tier"]) in {"low", "medium"}


def test_build_optimizer_operational_policy_includes_runtime_and_fallback_contract(monkeypatch):
    """The operational policy should make fallback, freshness, and runtime behavior explicit."""
    monkeypatch.setitem(module.MODELING_STAGE_PARALLEL["forecast_control"], "max_workers", 12)

    class DummyRuntimeSummary:
        def as_dict(self) -> dict[str, object]:
            return {
                "machine": "AMD64",
                "cpu_count": 24,
                "worker_cap": 12,
                "acceleration_mode": "auto",
                "xgboost": {
                    "available": True,
                    "device": "cuda",
                    "cuda_enabled": True,
                    "reason": "CUDA probe fit succeeded",
                },
            }

    monkeypatch.setattr(module, "runtime_summary", lambda configured_max_workers: DummyRuntimeSummary())
    policy = {
        "day_ahead": {"candidate_label": "day_ahead_model"},
        "hourly": {"candidate_label": "hourly_model"},
        "phase": {
            "candidate_label": "phase_model",
            "stack_guard_recommended_policy": "phase_candidate",
            "stack_guard_applied_candidate_label": "phase_stack_model",
        },
        "nowcast_anchor": {"candidate_label": "nowcast_model"},
        "day_ahead_refresh": {
            "recommended_policy": "triggered_refresh",
            "trigger_mode": "residual_or_activity",
            "refresh_interval_minutes": 60,
            "lookback_minutes": 120,
            "threshold_source": "rolling_benchmark",
            "residual_drift_mae_pct_threshold": 10.0,
            "transition_mae_pct_threshold": 20.0,
            "activity_ratio_shift_threshold": 0.1,
            "evaluation_trigger_rate": 0.25,
            "rolling_benchmark": {"trigger_rate": 0.3},
            "reason": "triggered refresh retained enough gain",
        },
    }
    uncertainty_summary = pd.DataFrame(
        [
            {
                "scope": "all_intervals",
                "interval_80_coverage": 0.9,
                "interval_95_coverage": 0.98,
            }
        ]
    )

    operational_policy = module._build_optimizer_operational_policy(
        run_id="run_456",
        config_hash="sha256:policy",
        policy=policy,
        lock_interval_minutes=15,
        uncertainty_summary=uncertainty_summary,
    )

    assert operational_policy["contract_version"] == module.OPTIMIZER_CONTRACT_VERSION
    assert operational_policy["layer_priority"] == module.OPTIMIZER_LAYER_PRIORITY
    assert (
        operational_policy["layer_contracts"]["nowcast"]["fallback_target_when_unavailable"] == "phase"
    )
    assert operational_policy["selection_policy"]["minute_layer_policy"]["stage10_operating_role"] == "corrective_overlay"
    assert operational_policy["uncertainty_policy"]["quantile_source_multipliers"]["lead_interval"] == 1.0
    assert operational_policy["uncertainty_policy"]["confidence_score_policy"]["layer_multipliers"]["nowcast"] == 1.0
    assert operational_policy["day_ahead_refresh_policy"]["recommended_policy"] == "triggered_refresh"
    assert operational_policy["hardware_policy"]["runtime_summary"]["worker_cap"] == 12
    assert operational_policy["hardware_policy"]["runtime_summary"]["xgboost"]["cuda_enabled"] is True


def test_parse_args_accepts_output_root_and_disable_replay_cache(monkeypatch, tmp_path):
    """CLI helpers should support isolated cold-run output roots without touching the default cache."""
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "forecast_control_backtest.py",
            "--output-root",
            str(tmp_path),
            "--disable-replay-cache",
        ],
    )

    args = module.parse_args()

    assert Path(args.output_root) == tmp_path
    assert bool(args.disable_replay_cache) is True


def test_benchmark_control_layer_candidates_can_promote_a_baseline_for_day_ahead():
    """Stage-10 should be able to choose the best replayed day-ahead baseline on control cycles."""
    by_origin = pd.DataFrame(
        [
            {
                "origin_timestamp": "2026-03-01T00:00:00",
                "candidate_label": "model::raw",
                "candidate_type": "learned",
                "source_model_label": "model",
                "target_mode": "raw",
                "endpoint_abs_error": 9.0,
                "endpoint_sq_error": 81.0,
                "endpoint_actual_abs": 100.0,
                "path_mae": 12.0,
                "path_rmse": 12.0,
                "path_abs_error_sum": 120.0,
                "path_actual_abs_sum": 1000.0,
                "phase_mean_abs_error": 8.0,
                "phase_mean_sq_error": 64.0,
                "phase_mean_actual_abs": 100.0,
                "next_lock_mae": 7.0,
                "next_lock_abs_error_sum": 70.0,
                "next_lock_actual_abs_sum": 500.0,
                "profile_shape_mae": 15.0,
                "profile_shape_abs_error_sum": 150.0,
                "profile_shape_actual_abs_sum": 1000.0,
                "energy_abs_error": 25.0,
                "energy_actual_abs": 1000.0,
                "coverage": 1.0,
            },
            {
                "origin_timestamp": "2026-03-01T00:00:00",
                "candidate_label": "hybrid_workday",
                "candidate_type": "baseline",
                "source_model_label": "hybrid_workday",
                "target_mode": "baseline",
                "endpoint_abs_error": 8.0,
                "endpoint_sq_error": 64.0,
                "endpoint_actual_abs": 100.0,
                "path_mae": 11.0,
                "path_rmse": 11.0,
                "path_abs_error_sum": 110.0,
                "path_actual_abs_sum": 1000.0,
                "phase_mean_abs_error": 7.0,
                "phase_mean_sq_error": 49.0,
                "phase_mean_actual_abs": 100.0,
                "next_lock_mae": 7.5,
                "next_lock_abs_error_sum": 75.0,
                "next_lock_actual_abs_sum": 500.0,
                "profile_shape_mae": 10.0,
                "profile_shape_abs_error_sum": 100.0,
                "profile_shape_actual_abs_sum": 1000.0,
                "energy_abs_error": 20.0,
                "energy_actual_abs": 1000.0,
                "coverage": 1.0,
            },
        ]
    )

    benchmark, winner = module._benchmark_control_layer_candidates(by_origin, layer_role="day_ahead")

    assert str(winner["candidate_label"]) == "hybrid_workday"
    assert str(winner["candidate_type"]) == "baseline"
    assert str(winner["selection_metric_name"]) == "profile_shape_mae"
    assert float(benchmark.iloc[0]["selection_metric_value"]) == 10.0
    assert float(benchmark.iloc[0]["profile_shape_mae_p50"]) == 10.0
    assert float(benchmark.iloc[0]["profile_shape_mae_p90"]) == 10.0


def test_benchmark_control_layer_candidates_can_rank_on_optimizer_score(monkeypatch):
    """Hourly/phase benchmarking should be able to use the optimizer composite score from detailed replay paths."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "hourly_selection_metric", "optimizer_score")
    by_origin = pd.DataFrame(
        [
            {
                "origin_timestamp": "2026-03-01T00:00:00",
                "candidate_label": "candidate_a",
                "candidate_type": "learned",
                "source_model_label": "model",
                "target_mode": "raw",
                "endpoint_abs_error": 5.0,
                "endpoint_sq_error": 25.0,
                "endpoint_actual_abs": 100.0,
                "path_mae": 5.0,
                "path_rmse": 5.0,
                "path_abs_error_sum": 50.0,
                "path_actual_abs_sum": 1000.0,
                "phase_mean_abs_error": 5.0,
                "phase_mean_sq_error": 25.0,
                "phase_mean_actual_abs": 100.0,
                "next_lock_mae": 5.0,
                "next_lock_abs_error_sum": 50.0,
                "next_lock_actual_abs_sum": 500.0,
                "profile_shape_mae": 5.0,
                "profile_shape_abs_error_sum": 50.0,
                "profile_shape_actual_abs_sum": 1000.0,
                "energy_abs_error": 5.0,
                "energy_actual_abs": 1000.0,
                "coverage": 1.0,
            },
            {
                "origin_timestamp": "2026-03-01T00:00:00",
                "candidate_label": "candidate_b",
                "candidate_type": "learned",
                "source_model_label": "model",
                "target_mode": "raw",
                "endpoint_abs_error": 5.0,
                "endpoint_sq_error": 25.0,
                "endpoint_actual_abs": 100.0,
                "path_mae": 5.0,
                "path_rmse": 5.0,
                "path_abs_error_sum": 50.0,
                "path_actual_abs_sum": 1000.0,
                "phase_mean_abs_error": 5.0,
                "phase_mean_sq_error": 25.0,
                "phase_mean_actual_abs": 100.0,
                "next_lock_mae": 5.0,
                "next_lock_abs_error_sum": 50.0,
                "next_lock_actual_abs_sum": 500.0,
                "profile_shape_mae": 5.0,
                "profile_shape_abs_error_sum": 50.0,
                "profile_shape_actual_abs_sum": 1000.0,
                "energy_abs_error": 5.0,
                "energy_actual_abs": 1000.0,
                "coverage": 1.0,
            },
        ]
    )
    detail_by_origin = pd.DataFrame(
        [
            {
                "origin_timestamp": "2026-03-01T00:00:00",
                "candidate_label": "candidate_a",
                "forecast_timestamp": "2026-03-01T00:05:00",
                "actual_load": 100.0,
                "predicted_load": 220.0,
            },
            {
                "origin_timestamp": "2026-03-01T00:00:00",
                "candidate_label": "candidate_a",
                "forecast_timestamp": "2026-03-01T00:10:00",
                "actual_load": 100.0,
                "predicted_load": 220.0,
            },
            {
                "origin_timestamp": "2026-03-01T00:00:00",
                "candidate_label": "candidate_a",
                "forecast_timestamp": "2026-03-01T00:15:00",
                "actual_load": 300.0,
                "predicted_load": 120.0,
            },
            {
                "origin_timestamp": "2026-03-01T00:00:00",
                "candidate_label": "candidate_b",
                "forecast_timestamp": "2026-03-01T00:05:00",
                "actual_load": 100.0,
                "predicted_load": 110.0,
            },
            {
                "origin_timestamp": "2026-03-01T00:00:00",
                "candidate_label": "candidate_b",
                "forecast_timestamp": "2026-03-01T00:10:00",
                "actual_load": 100.0,
                "predicted_load": 110.0,
            },
            {
                "origin_timestamp": "2026-03-01T00:00:00",
                "candidate_label": "candidate_b",
                "forecast_timestamp": "2026-03-01T00:15:00",
                "actual_load": 300.0,
                "predicted_load": 260.0,
            },
        ]
    )

    benchmark, winner = module._benchmark_control_layer_candidates(
        by_origin,
        layer_role="hourly",
        detail_by_origin=detail_by_origin,
    )

    assert str(winner["selection_metric_name"]) == "optimizer_score"
    assert str(winner["candidate_label"]) == "candidate_b"
    assert float(benchmark.loc[benchmark["candidate_label"].eq("candidate_b"), "peak_interval_hit_rate"].iloc[0]) == 1.0


def test_select_control_layer_candidate_can_prefer_held_out_evaluation(monkeypatch):
    """Stage-10 should be able to recommend the held-out winner instead of the calibration winner."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "optimize_replayed_candidates", True)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "control_promotion_scope", "held_out_evaluation")
    calibration = pd.DataFrame(
        [
            {
                "candidate_label": "calibration-best",
                "selection_metric_name": "next_lock_mae",
                "selection_metric_value": 100.0,
                "selection_metric_pct": 10.0,
            },
            {
                "candidate_label": "evaluation-best",
                "selection_metric_name": "next_lock_mae",
                "selection_metric_value": 105.0,
                "selection_metric_pct": 10.5,
            },
        ]
    )
    evaluation = pd.DataFrame(
        [
            {
                "candidate_label": "evaluation-best",
                "selection_metric_name": "next_lock_mae",
                "selection_metric_value": 95.0,
                "selection_metric_pct": 9.5,
            },
            {
                "candidate_label": "calibration-best",
                "selection_metric_name": "next_lock_mae",
                "selection_metric_value": 120.0,
                "selection_metric_pct": 12.0,
            },
        ]
    )

    selected, selection_mode = module._select_control_layer_candidate(
        calibration_benchmark=calibration,
        evaluation_benchmark=evaluation,
        upstream_label="calibration-best",
    )

    assert selection_mode == "held_out_control_layer_candidate_benchmark"
    assert str(selected["candidate_label"]) == "evaluation-best"


def test_select_control_layer_candidate_guard_can_keep_upstream_when_peak_regresses(monkeypatch):
    """Held-out promotion should keep the upstream choice when a challenger regresses next-lock/peak metrics."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "optimize_replayed_candidates", True)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "control_promotion_scope", "held_out_evaluation")
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "control_promotion_guard_enabled", True)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "control_promotion_guard_max_next_lock_regress_pct", 0.0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "control_promotion_guard_max_peak_value_regress_pct", 0.0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "control_promotion_guard_max_peak_miss_regress", 0.0)
    calibration = pd.DataFrame(
        [
            {"candidate_label": "upstream", "selection_metric_name": "optimizer_score", "selection_metric_value": 1.2},
            {"candidate_label": "challenger", "selection_metric_name": "optimizer_score", "selection_metric_value": 1.0},
        ]
    )
    evaluation = pd.DataFrame(
        [
            {
                "candidate_label": "challenger",
                "selection_metric_name": "optimizer_score",
                "selection_metric_value": 1.0,
                "next_lock_mae": 110.0,
                "peak_value_mae": 60.0,
                "peak_interval_miss_rate": 0.10,
            },
            {
                "candidate_label": "upstream",
                "selection_metric_name": "optimizer_score",
                "selection_metric_value": 1.1,
                "next_lock_mae": 100.0,
                "peak_value_mae": 50.0,
                "peak_interval_miss_rate": 0.00,
            },
        ]
    )

    selected, selection_mode = module._select_control_layer_candidate(
        calibration_benchmark=calibration,
        evaluation_benchmark=evaluation,
        upstream_label="upstream",
    )

    assert selection_mode == "held_out_control_layer_candidate_benchmark_guarded"
    assert str(selected["candidate_label"]) == "upstream"


def test_build_control_candidate_pool_adds_top_matching_rollout_registry_rows(monkeypatch):
    """Control replay should widen beyond the upstream selection using matching rollout-registry evidence."""
    registry = pd.DataFrame(
        [
            {
                "run_id": "run-best",
                "resolution": "10min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
                "horizon_minutes": 60,
                "origin_policy": "phase_balanced",
                "selection_target": "next_lock_mae",
                "strategy": "recursive",
                "learned_next_lock_mae": 250.0,
                "learned_origin_n": 8,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_next_lock": True,
                "generated_at_utc": "2026-03-11T01:00:00+00:00",
            },
            {
                "run_id": "run-second",
                "resolution": "5min",
                "feature_set": "curated",
                "model_label": "hgb-frontier",
                "horizon_minutes": 60,
                "origin_policy": "phase_balanced",
                "selection_target": "next_lock_mae",
                "strategy": "recursive",
                "learned_next_lock_mae": 275.0,
                "learned_origin_n": 8,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_next_lock": True,
                "generated_at_utc": "2026-03-11T00:00:00+00:00",
            },
            {
                "run_id": "wrong-origin",
                "resolution": "1min",
                "feature_set": "minimal",
                "model_label": "ignored",
                "horizon_minutes": 60,
                "origin_policy": "uniform",
                "selection_target": "next_lock_mae",
                "strategy": "recursive",
                "learned_next_lock_mae": 100.0,
                "learned_origin_n": 8,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_next_lock": True,
                "generated_at_utc": "2026-03-11T02:00:00+00:00",
            },
        ]
    )
    upstream_selection = {
        "resolution": "mixed",
        "feature_set": "portfolio",
        "model_label": "cross_candidate_portfolio",
        "portfolio_candidate_label": "cross_candidate_portfolio::phase_bucket_next_lock_policy",
    }
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "candidate_pool_size", 3)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "benchmark_expanded_candidate_pool_size", 3)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "benchmark_expanded_pool_layers", [])
    monkeypatch.setattr(module, "preferred_output_path", lambda _path: Path("outputs/007_rollout/commercial_facility"))
    monkeypatch.setattr(module, "_read_csv_if_present", lambda _path: registry.copy())
    monkeypatch.setattr(
        module,
        "resolve_rollout_selection_context",
        lambda **kwargs: {
            "resolution": kwargs["resolution"],
            "feature_set": kwargs["feature_set"],
            "model_label": kwargs["model_label"],
            "portfolio_candidate_label": "",
        },
    )

    pool = module._build_control_candidate_pool(
        layer_role="phase",
        upstream_selection=upstream_selection,
        horizon_minutes=60,
        origin_policy="phase_balanced",
        selection_target="next_lock_mae",
    )

    assert len(pool) == 3
    assert pool[0]["pool_source_type"] == "upstream_selection"
    assert pool[1]["selection"]["resolution"] == "10min"
    assert pool[2]["selection"]["resolution"] == "5min"


def test_build_control_candidate_pool_prioritizes_recent_phase_control_evidence(monkeypatch, tmp_path):
    """Phase replay should use recent Stage-10 evidence to rank challengers before exact replay."""
    registry = pd.DataFrame(
        [
            {
                "run_id": "run-poor-generic",
                "resolution": "1min",
                "feature_set": "curated",
                "model_label": "hgb-frontier",
                "horizon_minutes": 60,
                "origin_policy": "phase_balanced",
                "selection_target": "next_lock_mae",
                "strategy": "recursive",
                "learned_next_lock_mae": 100.0,
                "learned_origin_n": 8,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_next_lock": True,
                "generated_at_utc": "2026-03-11T03:00:00+00:00",
            },
            {
                "run_id": "run-mid-generic",
                "resolution": "1min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
                "horizon_minutes": 60,
                "origin_policy": "phase_balanced",
                "selection_target": "next_lock_mae",
                "strategy": "recursive",
                "learned_next_lock_mae": 150.0,
                "learned_origin_n": 8,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_next_lock": True,
                "generated_at_utc": "2026-03-11T02:00:00+00:00",
            },
            {
                "run_id": "run-best-control",
                "resolution": "5min",
                "feature_set": "full",
                "model_label": "hgb-frontier-lr010-leaf100",
                "horizon_minutes": 60,
                "origin_policy": "phase_balanced",
                "selection_target": "next_lock_mae",
                "strategy": "recursive",
                "learned_next_lock_mae": 300.0,
                "learned_origin_n": 8,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_next_lock": True,
                "generated_at_utc": "2026-03-11T01:00:00+00:00",
            },
        ]
    )
    forecast_root = tmp_path / "outputs" / "010_forecast_control" / "commercial_facility"
    for run_id in ("20260321T090000000000Z", "20260321T080000000000Z"):
        run_dir = forecast_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "control_layer": "phase",
                    "replay_resolution": "1min",
                    "replay_feature_set": "curated",
                    "replay_model_label": "hgb-frontier",
                    "candidate_label": "1min/curated/hgb-frontier::raw",
                    "evaluation_selection_metric_value": 50.0,
                    "evaluation_next_lock_mae": 500.0,
                    "evaluation_peak_value_mae": 600.0,
                    "evaluation_peak_interval_miss_rate": 0.50,
                },
                {
                    "control_layer": "phase",
                    "replay_resolution": "1min",
                    "replay_feature_set": "minimal",
                    "replay_model_label": "hgb-balanced",
                    "candidate_label": "1min/minimal/hgb-balanced::raw",
                    "evaluation_selection_metric_value": 35.0,
                    "evaluation_next_lock_mae": 350.0,
                    "evaluation_peak_value_mae": 450.0,
                    "evaluation_peak_interval_miss_rate": 0.25,
                },
                {
                    "control_layer": "phase",
                    "replay_resolution": "5min",
                    "replay_feature_set": "full",
                    "replay_model_label": "hgb-frontier-lr010-leaf100",
                    "candidate_label": "5min/full/hgb-frontier-lr010-leaf100::raw",
                    "evaluation_selection_metric_value": 34.0,
                    "evaluation_next_lock_mae": 340.0,
                    "evaluation_peak_value_mae": 440.0,
                    "evaluation_peak_interval_miss_rate": 0.20,
                },
            ]
        ).to_csv(run_dir / "control_layer_candidate_benchmarks.csv", index=False)

    upstream_selection = {
        "resolution": "mixed",
        "feature_set": "portfolio",
        "model_label": "cross_candidate_portfolio",
        "portfolio_candidate_label": "cross_candidate_portfolio::phase_bucket_next_lock_policy",
    }

    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_control_candidate_pool_size", 3)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_control_prior_run_limit", 2)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_control_min_prior_support_runs", 2)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_control_max_supplemental_contexts_per_resolution", 1)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_control_exploration_slots", 1)
    monkeypatch.setattr(
        module,
        "preferred_output_path",
        lambda path: forecast_root if "010_forecast_control" in str(path) else tmp_path / "outputs" / "007_rollout" / "commercial_facility",
    )

    def _fake_read_csv(path: Path) -> pd.DataFrame:
        path = Path(path)
        if path.name == "rollout_registry.csv":
            return registry.copy()
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    monkeypatch.setattr(module, "_read_csv_if_present", _fake_read_csv)
    monkeypatch.setattr(
        module,
        "resolve_rollout_selection_context",
        lambda **kwargs: {
            "resolution": kwargs["resolution"],
            "feature_set": kwargs["feature_set"],
            "model_label": kwargs["model_label"],
            "portfolio_candidate_label": "",
        },
    )

    pool = module._build_control_candidate_pool(
        layer_role="phase",
        upstream_selection=upstream_selection,
        horizon_minutes=60,
        origin_policy="phase_balanced",
        selection_target="next_lock_mae",
    )

    assert len(pool) == 3
    assert pool[1]["selection"]["resolution"] == "5min"
    assert pool[1]["selection"]["feature_set"] == "full"
    assert pool[1]["pool_prior_phase_support_runs"] == 2
    assert pool[2]["selection"]["feature_set"] == "minimal"


def test_build_control_candidate_pool_filters_phase_contexts_by_prior_support_and_resolution(monkeypatch, tmp_path):
    """Phase replay should keep prior-backed diversity first and reserve one exploration slot."""
    registry = pd.DataFrame(
        [
            {
                "run_id": "run-1min-best",
                "resolution": "1min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
                "horizon_minutes": 60,
                "origin_policy": "phase_balanced",
                "selection_target": "next_lock_mae",
                "strategy": "recursive",
                "learned_next_lock_mae": 120.0,
                "learned_origin_n": 8,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_next_lock": True,
                "generated_at_utc": "2026-03-11T05:00:00+00:00",
            },
            {
                "run_id": "run-1min-second",
                "resolution": "1min",
                "feature_set": "curated",
                "model_label": "hgb-frontier",
                "horizon_minutes": 60,
                "origin_policy": "phase_balanced",
                "selection_target": "next_lock_mae",
                "strategy": "recursive",
                "learned_next_lock_mae": 125.0,
                "learned_origin_n": 8,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_next_lock": True,
                "generated_at_utc": "2026-03-11T04:00:00+00:00",
            },
            {
                "run_id": "run-5min-best",
                "resolution": "5min",
                "feature_set": "full",
                "model_label": "hgb-frontier-lr010-leaf100",
                "horizon_minutes": 60,
                "origin_policy": "phase_balanced",
                "selection_target": "next_lock_mae",
                "strategy": "recursive",
                "learned_next_lock_mae": 140.0,
                "learned_origin_n": 8,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_next_lock": True,
                "generated_at_utc": "2026-03-11T03:00:00+00:00",
            },
            {
                "run_id": "run-explore",
                "resolution": "10min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
                "horizon_minutes": 60,
                "origin_policy": "phase_balanced",
                "selection_target": "next_lock_mae",
                "strategy": "recursive",
                "learned_next_lock_mae": 155.0,
                "learned_origin_n": 8,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_next_lock": True,
                "generated_at_utc": "2026-03-11T02:00:00+00:00",
            },
        ]
    )
    forecast_root = tmp_path / "outputs" / "010_forecast_control" / "commercial_facility"
    run_dir = forecast_root / "20260321T090000000000Z"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "control_layer": "phase",
                "replay_resolution": "1min",
                "replay_feature_set": "minimal",
                "replay_model_label": "hgb-balanced",
                "candidate_label": "1min/minimal/hgb-balanced::raw",
                "evaluation_selection_metric_value": 30.0,
                "evaluation_next_lock_mae": 300.0,
                "evaluation_peak_value_mae": 400.0,
                "evaluation_peak_interval_miss_rate": 0.20,
            },
            {
                "control_layer": "phase",
                "replay_resolution": "1min",
                "replay_feature_set": "curated",
                "replay_model_label": "hgb-frontier",
                "candidate_label": "1min/curated/hgb-frontier::raw",
                "evaluation_selection_metric_value": 32.0,
                "evaluation_next_lock_mae": 320.0,
                "evaluation_peak_value_mae": 420.0,
                "evaluation_peak_interval_miss_rate": 0.25,
            },
            {
                "control_layer": "phase",
                "replay_resolution": "5min",
                "replay_feature_set": "full",
                "replay_model_label": "hgb-frontier-lr010-leaf100",
                "candidate_label": "5min/full/hgb-frontier-lr010-leaf100::raw",
                "evaluation_selection_metric_value": 31.0,
                "evaluation_next_lock_mae": 310.0,
                "evaluation_peak_value_mae": 410.0,
                "evaluation_peak_interval_miss_rate": 0.22,
            },
        ]
    ).to_csv(run_dir / "control_layer_candidate_benchmarks.csv", index=False)

    upstream_selection = {
        "resolution": "mixed",
        "feature_set": "portfolio",
        "model_label": "cross_candidate_portfolio",
        "portfolio_candidate_label": "cross_candidate_portfolio::phase_bucket_next_lock_policy",
    }

    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_control_candidate_pool_size", 4)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_control_prior_run_limit", 1)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_control_min_prior_support_runs", 1)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_control_max_supplemental_contexts_per_resolution", 1)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_control_exploration_slots", 1)
    monkeypatch.setattr(
        module,
        "preferred_output_path",
        lambda path: forecast_root if "010_forecast_control" in str(path) else tmp_path / "outputs" / "007_rollout" / "commercial_facility",
    )

    def _fake_read_csv(path: Path) -> pd.DataFrame:
        path = Path(path)
        if path.name == "rollout_registry.csv":
            return registry.copy()
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    monkeypatch.setattr(module, "_read_csv_if_present", _fake_read_csv)
    monkeypatch.setattr(
        module,
        "resolve_rollout_selection_context",
        lambda **kwargs: {
            "resolution": kwargs["resolution"],
            "feature_set": kwargs["feature_set"],
            "model_label": kwargs["model_label"],
            "portfolio_candidate_label": "",
        },
    )

    pool = module._build_control_candidate_pool(
        layer_role="phase",
        upstream_selection=upstream_selection,
        horizon_minutes=60,
        origin_policy="phase_balanced",
        selection_target="next_lock_mae",
    )

    assert len(pool) == 4
    assert pool[1]["selection"]["resolution"] == "1min"
    assert pool[1]["selection"]["feature_set"] == "minimal"
    assert pool[2]["selection"]["resolution"] == "5min"
    assert pool[3]["selection"]["resolution"] == "10min"
    assert "exploration slot" in str(pool[3]["pool_reason"]).lower()


def test_build_control_candidate_pool_relaxes_phase_pruning_when_no_priors_exist(monkeypatch):
    """Phase replay should not collapse to one challenger when no prior evidence exists yet."""
    registry = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "resolution": "1min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
                "horizon_minutes": 60,
                "origin_policy": "phase_balanced",
                "selection_target": "next_lock_mae",
                "strategy": "recursive",
                "learned_next_lock_mae": 120.0,
                "learned_origin_n": 8,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_next_lock": True,
                "generated_at_utc": "2026-03-11T05:00:00+00:00",
            },
            {
                "run_id": "run-2",
                "resolution": "5min",
                "feature_set": "full",
                "model_label": "hgb-frontier-lr010-leaf100",
                "horizon_minutes": 60,
                "origin_policy": "phase_balanced",
                "selection_target": "next_lock_mae",
                "strategy": "recursive",
                "learned_next_lock_mae": 130.0,
                "learned_origin_n": 8,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_next_lock": True,
                "generated_at_utc": "2026-03-11T04:00:00+00:00",
            },
            {
                "run_id": "run-3",
                "resolution": "10min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
                "horizon_minutes": 60,
                "origin_policy": "phase_balanced",
                "selection_target": "next_lock_mae",
                "strategy": "recursive",
                "learned_next_lock_mae": 140.0,
                "learned_origin_n": 8,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_next_lock": True,
                "generated_at_utc": "2026-03-11T03:00:00+00:00",
            },
        ]
    )
    upstream_selection = {
        "resolution": "mixed",
        "feature_set": "portfolio",
        "model_label": "cross_candidate_portfolio",
        "portfolio_candidate_label": "cross_candidate_portfolio::phase_bucket_next_lock_policy",
    }

    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_control_candidate_pool_size", 4)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_control_min_prior_support_runs", 2)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_control_max_supplemental_contexts_per_resolution", 1)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_control_exploration_slots", 1)
    monkeypatch.setattr(module, "preferred_output_path", lambda path: Path("D:/missing") if "010_forecast_control" in str(path) else Path("D:/other"))
    monkeypatch.setattr(module, "_read_csv_if_present", lambda path: registry.copy() if Path(path).name == "rollout_registry.csv" else pd.DataFrame())
    monkeypatch.setattr(
        module,
        "resolve_rollout_selection_context",
        lambda **kwargs: {
            "resolution": kwargs["resolution"],
            "feature_set": kwargs["feature_set"],
            "model_label": kwargs["model_label"],
            "portfolio_candidate_label": "",
        },
    )

    pool = module._build_control_candidate_pool(
        layer_role="phase",
        upstream_selection=upstream_selection,
        horizon_minutes=60,
        origin_policy="phase_balanced",
        selection_target="next_lock_mae",
    )

    assert len(pool) == 4
    assert [item["selection"]["resolution"] for item in pool[1:]] == ["1min", "5min", "10min"]


def test_control_candidate_pool_size_expands_for_configured_intraday_layers(monkeypatch):
    """Intraday control layers should be able to widen their benchmark pool without changing day-ahead scope."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "candidate_pool_size", 2)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "benchmark_expanded_candidate_pool_size", 6)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "benchmark_expanded_pool_layers", ["hourly", "phase"])
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_control_candidate_pool_size", 4)

    assert module._control_candidate_pool_size("day_ahead") == 2
    assert module._control_candidate_pool_size("hourly") == 6
    assert module._control_candidate_pool_size("phase") == 4


def test_control_benchmark_origins_can_use_full_exact_scope_for_intraday_layers(monkeypatch):
    """Configured intraday layers should benchmark on the full exact-control origin set."""
    origins = [pd.Timestamp("2026-03-01T00:00:00") + pd.Timedelta(minutes=15 * idx) for idx in range(6)]
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "candidate_benchmark_origin_cap", 2)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "benchmark_full_origin_layers", ["hourly", "phase"])

    day_ahead_origins, day_ahead_mode = module._control_benchmark_origins(
        layer_role="day_ahead",
        origin_timestamps=origins,
    )
    phase_origins, phase_mode = module._control_benchmark_origins(
        layer_role="phase",
        origin_timestamps=origins,
    )

    assert len(day_ahead_origins) == 2
    assert day_ahead_mode == "sampled_cap"
    assert phase_origins == origins
    assert phase_mode == "full_control_scope"


def test_run_cached_rollout_evaluation_reuses_exact_origin_cache(monkeypatch, tmp_path):
    """Exact-origin replay cache should serve repeated control-layer requests without recomputation."""
    selection = {
        "resolution": "10min",
        "feature_set": "minimal",
        "model_label": "hgb-balanced",
        "selection_policy": "registry",
        "selection_source": "outputs/007_rollout/commercial_facility/rollout_registry.csv",
    }
    call_count = 0

    def _fake_run_rollout_evaluation(**kwargs):
        nonlocal call_count
        call_count += 1
        origin_timestamps = list(kwargs["origin_timestamps"])
        by_origin = pd.DataFrame(
            [
                {
                    "origin_timestamp": pd.Timestamp(origin_timestamps[0]).isoformat(),
                    "candidate_label": "hgb-balanced::raw",
                    "candidate_type": "learned",
                    "source_model_label": "hgb-balanced",
                    "target_mode": "raw",
                    "endpoint_abs_error": 5.0,
                    "endpoint_sq_error": 25.0,
                    "endpoint_actual_abs": 100.0,
                    "path_mae": 6.0,
                    "path_rmse": 6.0,
                    "path_abs_error_sum": 60.0,
                    "path_actual_abs_sum": 1000.0,
                    "phase_mean_abs_error": 4.0,
                    "phase_mean_sq_error": 16.0,
                    "phase_mean_actual_abs": 100.0,
                    "next_lock_mae": 3.0,
                    "next_lock_abs_error_sum": 30.0,
                    "next_lock_actual_abs_sum": 500.0,
                    "profile_shape_mae": 7.0,
                    "profile_shape_abs_error_sum": 70.0,
                    "profile_shape_actual_abs_sum": 1000.0,
                    "energy_abs_error": 8.0,
                    "energy_actual_abs": 1000.0,
                    "coverage": 1.0,
                    "n_eval": 6.0,
                }
            ]
        )
        detail_by_origin = pd.DataFrame(
            [
                {
                    "origin_timestamp": pd.Timestamp(origin_timestamps[0]).isoformat(),
                    "candidate_label": "hgb-balanced::raw",
                    "forecast_timestamp": (
                        pd.Timestamp(origin_timestamps[0]) + pd.Timedelta(minutes=10)
                    ).isoformat(),
                    "predicted_load": 100.0,
                }
            ]
        )
        selected_origins = pd.DataFrame(
            [
                {
                    "origin_position": 1,
                    "origin_timestamp": pd.Timestamp(origin_timestamps[0]).isoformat(),
                }
            ]
        )
        metrics = pd.DataFrame([{"candidate_label": "hgb-balanced::raw", "path_mae": 6.0}])
        selection_summary = pd.DataFrame(
            [{"selection_target": "next_lock_mae", "winner_candidate_label": "hgb-balanced::raw"}]
        )
        rollout_health = pd.DataFrame([{"status": "pass"}])
        return {
            "run_dir": tmp_path / "ephemeral_stage7",
            "metrics": metrics,
            "by_origin": by_origin,
            "selected_origins": selected_origins,
            "selection_summary": selection_summary,
            "rollout_health": rollout_health,
            "manifest": {"run_id": "ephemeral"},
            "selection": selection,
            "detail_by_origin": detail_by_origin,
        }

    monkeypatch.setattr(module, "run_rollout_evaluation", _fake_run_rollout_evaluation)

    first = module._run_cached_rollout_evaluation(
        cache_root=tmp_path / "replay_cache",
        temp_output_root=tmp_path / "temp_stage7",
        layer_role="hourly",
        selection=selection,
        horizon_minutes=60,
        origin_policy="phase_balanced",
        selection_target="next_lock_mae",
        origin_timestamps=[pd.Timestamp("2026-03-01T00:00:00")],
        capture_path_details=True,
        candidate_scope="selected_only",
        persist_artifacts=True,
    )
    second = module._run_cached_rollout_evaluation(
        cache_root=tmp_path / "replay_cache",
        temp_output_root=tmp_path / "temp_stage7",
        layer_role="hourly",
        selection=selection,
        horizon_minutes=60,
        origin_policy="phase_balanced",
        selection_target="next_lock_mae",
        origin_timestamps=[pd.Timestamp("2026-03-01T00:00:00")],
        capture_path_details=True,
        candidate_scope="selected_only",
        persist_artifacts=True,
    )

    assert call_count == 1
    assert str(first["replay_cache_status"]) == "miss"
    assert str(second["replay_cache_status"]) == "hit"
    assert Path(second["run_dir"]).exists()
    assert (tmp_path / "replay_cache" / "replay_cache_registry.csv").exists()
    assert second["detail_by_origin"].equals(first["detail_by_origin"])


def test_run_cached_rollout_evaluation_can_reuse_covering_origin_subset_cache(monkeypatch, tmp_path):
    """A larger exact-origin replay should satisfy later subset-origin requests."""
    selection = {
        "resolution": "10min",
        "feature_set": "minimal",
        "model_label": "hgb-balanced",
        "selection_policy": "registry",
        "selection_source": "outputs/007_rollout/commercial_facility/rollout_registry.csv",
        "requested_candidate_label": "hgb-balanced::raw",
    }
    call_count = 0

    def _fake_run_rollout_evaluation(**kwargs):
        nonlocal call_count
        call_count += 1
        origin_timestamps = [pd.Timestamp(value) for value in kwargs["origin_timestamps"]]
        by_origin = pd.DataFrame(
            [
                {
                    "origin_timestamp": timestamp.isoformat(),
                    "candidate_label": "hgb-balanced::raw",
                    "candidate_type": "learned",
                    "source_model_label": "hgb-balanced",
                    "target_mode": "raw",
                    "endpoint_abs_error": 5.0,
                    "endpoint_sq_error": 25.0,
                    "endpoint_actual_abs": 100.0,
                    "path_mae": 6.0,
                    "path_rmse": 6.0,
                    "path_abs_error_sum": 60.0,
                    "path_actual_abs_sum": 1000.0,
                    "phase_mean_abs_error": 4.0,
                    "phase_mean_sq_error": 16.0,
                    "phase_mean_actual_abs": 100.0,
                    "next_lock_mae": 3.0,
                    "next_lock_abs_error_sum": 30.0,
                    "next_lock_actual_abs_sum": 500.0,
                    "profile_shape_mae": 7.0,
                    "profile_shape_abs_error_sum": 70.0,
                    "profile_shape_actual_abs_sum": 1000.0,
                    "energy_abs_error": 8.0,
                    "energy_actual_abs": 1000.0,
                    "coverage": 1.0,
                    "n_eval": 6.0,
                }
                for timestamp in origin_timestamps
            ]
        )
        detail_by_origin = pd.DataFrame(
            [
                {
                    "origin_timestamp": timestamp.isoformat(),
                    "candidate_label": "hgb-balanced::raw",
                    "forecast_timestamp": (timestamp + pd.Timedelta(minutes=10)).isoformat(),
                    "predicted_load": 100.0,
                }
                for timestamp in origin_timestamps
            ]
        )
        selected_origins = pd.DataFrame(
            [
                {
                    "origin_position": idx + 1,
                    "origin_timestamp": timestamp.isoformat(),
                }
                for idx, timestamp in enumerate(origin_timestamps)
            ]
        )
        return {
            "run_dir": tmp_path / "ephemeral_stage7",
            "metrics": pd.DataFrame([{"candidate_label": "hgb-balanced::raw", "path_mae": 6.0}]),
            "by_origin": by_origin,
            "selected_origins": selected_origins,
            "selection_summary": pd.DataFrame(
                [{"selection_target": "next_lock_mae", "winner_candidate_label": "hgb-balanced::raw"}]
            ),
            "rollout_health": pd.DataFrame([{"status": "pass"}]),
            "manifest": {"run_id": "ephemeral"},
            "selection": selection,
            "detail_by_origin": detail_by_origin,
        }

    monkeypatch.setattr(module, "run_rollout_evaluation", _fake_run_rollout_evaluation)
    origin_a = pd.Timestamp("2026-03-01T00:00:00")
    origin_b = pd.Timestamp("2026-03-01T01:00:00")

    first = module._run_cached_rollout_evaluation(
        cache_root=tmp_path / "replay_cache",
        temp_output_root=tmp_path / "temp_stage7",
        layer_role="phase",
        selection=selection,
        horizon_minutes=15,
        origin_policy="phase_balanced",
        selection_target="next_lock_mae",
        origin_timestamps=[origin_a, origin_b],
        capture_path_details=True,
        candidate_scope="selected_only",
        persist_artifacts=True,
    )
    second = module._run_cached_rollout_evaluation(
        cache_root=tmp_path / "replay_cache",
        temp_output_root=tmp_path / "temp_stage7",
        layer_role="phase",
        selection=selection,
        horizon_minutes=15,
        origin_policy="phase_balanced",
        selection_target="next_lock_mae",
        origin_timestamps=[origin_a],
        capture_path_details=True,
        candidate_scope="selected_only",
        persist_artifacts=True,
    )

    assert call_count == 1
    assert str(first["replay_cache_status"]) == "miss"
    assert str(second["replay_cache_status"]) == "subset_hit"
    assert list(second["selected_origins"]["origin_timestamp"]) == [origin_a.isoformat()]
    assert set(second["by_origin"]["origin_timestamp"].astype("string")) == {origin_a.isoformat()}
    assert set(second["detail_by_origin"]["origin_timestamp"].astype("string")) == {origin_a.isoformat()}


def test_replay_rollout_layer_can_promote_an_alternative_control_candidate(monkeypatch, tmp_path):
    """Control replay should compare a small candidate pool on shared cycles, not only the upstream run."""
    monkeypatch.setattr(module, "resolve_rollout_origin_policy", lambda *_args, **_kwargs: "phase_balanced")
    monkeypatch.setattr(module, "resolve_rollout_selection_target", lambda *_args, **_kwargs: "next_lock_mae")
    monkeypatch.setattr(
        module,
        "_shared_representable_origins",
        lambda **kwargs: list(kwargs["origin_timestamps"]),
    )
    upstream_selection = {
        "resolution": "mixed",
        "feature_set": "portfolio",
        "model_label": "cross_candidate_portfolio",
        "portfolio_candidate_label": "cross_candidate_portfolio::phase_bucket_next_lock_policy",
    }
    monkeypatch.setattr(module, "resolve_rollout_selection_context", lambda **_kwargs: dict(upstream_selection))
    alternative_selection = {
        "resolution": "10min",
        "feature_set": "minimal",
        "model_label": "hgb-balanced",
        "portfolio_candidate_label": "",
    }
    monkeypatch.setattr(
        module,
        "_build_control_candidate_pool",
        lambda **_kwargs: [
            {"selection": dict(upstream_selection), "pool_source_type": "upstream_selection", "pool_source_run_id": "", "pool_reason": ""},
            {"selection": dict(alternative_selection), "pool_source_type": "rollout_registry", "pool_source_run_id": "run-best", "pool_reason": ""},
        ],
    )
    replay_flags: list[tuple[str, bool, bool, str]] = []

    def _fake_result(selection: dict[str, str]) -> dict[str, object]:
        native_label = (
            "cross_candidate_portfolio::phase_bucket_next_lock_policy"
            if selection["resolution"] == "mixed"
            else "hgb-balanced::raw"
        )
        next_lock = 6.0 if selection["resolution"] == "mixed" else 4.0
        by_origin = pd.DataFrame(
            [
                {
                    "origin_timestamp": "2026-03-01T00:00:00",
                    "candidate_label": native_label,
                    "candidate_type": "learned",
                    "source_model_label": selection["model_label"],
                    "target_mode": "raw",
                    "endpoint_abs_error": next_lock + 2.0,
                    "endpoint_sq_error": (next_lock + 2.0) ** 2,
                    "endpoint_actual_abs": 100.0,
                    "path_mae": next_lock + 1.0,
                    "path_rmse": next_lock + 1.0,
                    "path_abs_error_sum": (next_lock + 1.0) * 10.0,
                    "path_actual_abs_sum": 1000.0,
                    "phase_mean_abs_error": next_lock + 0.5,
                    "phase_mean_sq_error": (next_lock + 0.5) ** 2,
                    "phase_mean_actual_abs": 100.0,
                    "next_lock_mae": next_lock,
                    "next_lock_abs_error_sum": next_lock * 5.0,
                    "next_lock_actual_abs_sum": 500.0,
                    "profile_shape_mae": next_lock + 3.0,
                    "profile_shape_abs_error_sum": (next_lock + 3.0) * 10.0,
                    "profile_shape_actual_abs_sum": 1000.0,
                    "energy_abs_error": next_lock + 4.0,
                    "energy_actual_abs": 1000.0,
                    "coverage": 1.0,
                    "n_eval": 4.0,
                }
            ]
        )
        detail = pd.DataFrame(
            [
                {
                    "origin_timestamp": "2026-03-01T00:00:00",
                    "candidate_label": native_label,
                    "forecast_timestamp": "2026-03-01T00:15:00",
                    "predicted_load": 100.0,
                }
            ]
        )
        return {
            "run_dir": tmp_path / selection["resolution"],
            "by_origin": by_origin,
            "detail_by_origin": detail,
        }

    monkeypatch.setattr(
        module,
        "run_rollout_evaluation",
        lambda *, selection, capture_path_details, persist_artifacts, candidate_scope, **_kwargs: (
            replay_flags.append(
                (
                    selection["resolution"],
                    bool(capture_path_details),
                    bool(persist_artifacts),
                    str(candidate_scope),
                )
            ),
            _fake_result(selection),
        )[1],
    )

    replay = module._replay_rollout_layer(
        temp_root=tmp_path / "control_pool",
        cache_root=None,
        layer_role="hourly",
        horizon_minutes=60,
        origin_timestamps=[pd.Timestamp("2026-03-01T00:00:00")],
    )

    assert replay["candidate_pool_count"] == 2
    assert replay["upstream_candidate_label"] == (
        "mixed/portfolio/cross_candidate_portfolio::phase_bucket_next_lock_policy"
    )
    assert replay["candidate_label"] == "10min/minimal/hgb-balanced::raw"
    assert replay["selection"]["resolution"] == "10min"
    assert replay["selection"]["requested_candidate_label"] == "hgb-balanced::raw"
    assert replay["result"]["detail_by_origin"]["candidate_label"].astype("string").str.contains(
        "10min/minimal/hgb-balanced::raw"
    ).any()
    assert sorted(replay_flags[:2]) == sorted(
        [
            ("mixed", False, False, "selected_plus_baselines"),
            ("10min", False, False, "selected_plus_baselines"),
        ]
    )
    assert replay_flags[2:] == [
        ("10min", True, False, "selected_only"),
        ("10min", True, True, "selected_only"),
    ]
    assert replay["selected_replay_cache_status"] == "disabled"
    assert replay["benchmark_origin_mode"] == "full_control_scope"


def test_write_summary_md_embeds_visuals_and_cache_context(tmp_path):
    """Stage-10 markdown should explain the figures and show embedded image links."""
    summary = pd.DataFrame(
        [
            {
                "layer": "day_ahead",
                "minute_path_mae": 10.0,
                "minute_path_mae_pct": 5.0,
                "lock_mae": 8.0,
                "lock_mae_pct": 4.0,
                "profile_shape_mae": 7.0,
                "profile_shape_mae_pct": 3.5,
                "energy_mae": 6.0,
                "energy_mae_pct": 3.0,
                "lock_mae_gain_vs_day_ahead": 0.0,
                "cycle_n": 2,
            }
        ]
    )
    policy = {
        "day_ahead": {
            "candidate_label": "10min/minimal/hybrid_workday",
            "candidate_type": "baseline",
            "control_selection_metric": "profile_shape_mae",
            "control_selection_metric_value": 7.0,
            "control_selection_metric_pct": 3.5,
            "benchmark_origin_mode": "sampled_cap",
            "candidate_pool_count": 2,
            "selected_replay_cache_status": "hit",
            "selected_replay_cache_artifact": "outputs/010_forecast_control/commercial_facility/replay_cache/day_ahead/demo",
        },
        "hourly": {
            "candidate_label": "10min/minimal/hgb-balanced::raw",
            "candidate_type": "learned",
            "control_selection_metric": "next_lock_mae",
            "control_selection_metric_value": 5.0,
            "control_selection_metric_pct": 2.5,
            "benchmark_origin_mode": "full_control_scope",
            "candidate_pool_count": 6,
            "selected_replay_cache_status": "miss",
            "selected_replay_cache_artifact": "outputs/010_forecast_control/commercial_facility/replay_cache/hourly/demo",
        },
        "phase": {
            "candidate_label": "5min/minimal/hgb-balanced::phase_bucket_next_lock_policy",
            "candidate_type": "learned",
            "control_selection_metric": "next_lock_mae",
            "control_selection_metric_value": 4.0,
            "control_selection_metric_pct": 2.0,
            "benchmark_origin_mode": "full_control_scope",
            "candidate_pool_count": 6,
            "selected_replay_cache_status": "hit",
            "selected_replay_cache_artifact": "outputs/010_forecast_control/commercial_facility/replay_cache/phase/demo",
            "selection_artifact": "outputs/010_forecast_control/commercial_facility/replay_cache/phase/demo",
        },
        "nowcast_anchor": {
            "candidate_label": "persistence",
            "candidate_type": "baseline",
            "control_selection_metric": "lock_mae",
            "control_selection_metric_value": 2.0,
            "control_selection_metric_pct": 1.0,
            "benchmark_origin_mode": "full_control_scope",
            "candidate_pool_count": 7,
        },
        "day_ahead_refresh": {
            "candidate_label": "hgb-balanced::hybrid_workday_residual",
            "refresh_interval_minutes": 60,
            "recommended_policy": "triggered_refresh",
            "reason": "Triggered refresh improved profile-shape without a lock regression.",
        },
    }
    refresh_summary = pd.DataFrame(
        [
            {
                "scenario": "frozen_day_ahead",
                "minute_path_mae": 10.0,
                "minute_path_mae_pct": 5.0,
                "lock_mae": 8.0,
                "lock_mae_pct": 4.0,
                "profile_shape_mae": 7.0,
                "profile_shape_mae_pct": 3.5,
                "energy_mae": 6.0,
                "energy_mae_pct": 3.0,
                "lock_mae_gain_vs_frozen": 0.0,
                "profile_shape_mae_gain_vs_frozen": 0.0,
                "refresh_update_count": 0.0,
            },
            {
                "scenario": "triggered_refresh",
                "minute_path_mae": 9.0,
                "minute_path_mae_pct": 4.5,
                "lock_mae": 7.5,
                "lock_mae_pct": 3.8,
                "profile_shape_mae": 6.0,
                "profile_shape_mae_pct": 3.0,
                "energy_mae": 5.0,
                "energy_mae_pct": 2.5,
                "lock_mae_gain_vs_frozen": 0.5,
                "profile_shape_mae_gain_vs_frozen": 1.0,
                "refresh_update_count": 2.0,
            },
        ]
    )

    module._write_summary_md(
        output_dir=tmp_path,
        summary=summary,
        policy=policy,
        refresh_summary=refresh_summary,
    )

    text = (tmp_path / "control_backtest_summary.md").read_text(encoding="utf-8")
    assert "## Visuals" in text
    assert "![Locked-interval MAE progression](fig_control_lock_mae.png)" in text
    assert "![Example 24h control cycle](fig_control_example_cycle.png)" in text
    assert "## Day-Ahead Refresh Study" in text
    assert "![Day-ahead refresh policy comparison](fig_day_ahead_refresh_policy.png)" in text
    assert "Day-ahead replay cache" in text
    assert "1-minute nowcast policy" in text
    assert "after_nowcast" in text


def test_day_ahead_refresh_decision_row_triggers_on_residual_drift_and_activity_shift(monkeypatch):
    """Refresh triggers should fire when the frozen profile drifts materially from realized load."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "day_ahead_refresh_lookback_minutes", 60)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "day_ahead_refresh_residual_drift_mae_pct_threshold", 5.0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "day_ahead_refresh_transition_mae_pct_threshold", 5.0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "day_ahead_refresh_activity_ratio_shift_threshold", 0.05)
    minute_index = pd.date_range("2026-03-01T01:01:00", periods=60, freq="1min")
    feature_frame = pd.DataFrame(
        {
            "timestamp": minute_index,
            "avg_load": [150.0] * 60,
            "workday_transition": [0.0] * 60,
            "avg_workday_baseline": [100.0] * 60,
            "profile_activity_ratio": [0.40] * 60,
        }
    ).set_index("timestamp", drop=False)
    frozen = pd.Series([100.0] * 60, index=minute_index, dtype=float)

    decision = module._day_ahead_refresh_decision_row(
        cycle_origin_timestamp=pd.Timestamp("2026-03-01T00:00:00"),
        refresh_origin_timestamp=pd.Timestamp("2026-03-01T02:00:00"),
        minute_feature_frame=feature_frame,
        frozen_forecast=frozen,
    )

    assert bool(decision["refresh_triggered"]) is True
    assert bool(decision["residual_drift_trigger"]) is True
    assert bool(decision["activity_profile_shift_trigger"]) is True


def test_recommend_day_ahead_refresh_promotes_triggered_only_when_profile_and_lock_improve(monkeypatch):
    """Triggered refresh should only be promoted when it beats the frozen path on the exact-cycle rule."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "day_ahead_refresh_min_trigger_rate", 0.1)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "day_ahead_refresh_max_trigger_rate", 0.9)
    refresh_summary = pd.DataFrame(
        [
            {
                "scenario": "frozen_day_ahead",
                "lock_mae": 10.0,
                "profile_shape_mae": 12.0,
            },
            {
                "scenario": "unconditional_refresh",
                "lock_mae": 9.0,
                "profile_shape_mae": 11.0,
            },
            {
                "scenario": "triggered_refresh",
                "lock_mae": 8.5,
                "profile_shape_mae": 10.0,
            },
        ]
    )
    refresh_decisions = pd.DataFrame({"refresh_triggered": [True, False, True]})

    recommendation = module._recommend_day_ahead_refresh(refresh_summary, refresh_decisions)

    assert recommendation["recommended_policy"] == "triggered_refresh"
    assert bool(recommendation["triggered_beats_frozen_profile_shape"]) is True
    assert bool(recommendation["triggered_beats_frozen_lock"]) is True


def test_recommend_day_ahead_refresh_falls_back_to_unconditional_when_trigger_is_not_selective(monkeypatch):
    """A useful but always-on trigger should be reported as unconditional refresh, not selective promotion."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "day_ahead_refresh_min_trigger_rate", 0.1)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "day_ahead_refresh_max_trigger_rate", 0.8)
    refresh_summary = pd.DataFrame(
        [
            {
                "scenario": "frozen_day_ahead",
                "lock_mae": 10.0,
                "profile_shape_mae": 12.0,
            },
            {
                "scenario": "unconditional_refresh",
                "lock_mae": 9.0,
                "profile_shape_mae": 11.0,
            },
            {
                "scenario": "triggered_refresh",
                "lock_mae": 9.0,
                "profile_shape_mae": 11.0,
            },
        ]
    )
    refresh_decisions = pd.DataFrame({"refresh_triggered": [True, True, True, True]})

    recommendation = module._recommend_day_ahead_refresh(refresh_summary, refresh_decisions)

    assert recommendation["recommended_policy"] == "unconditional_refresh"
    assert bool(recommendation["trigger_rate_in_band"]) is False


def test_recommend_day_ahead_refresh_falls_back_to_unconditional_when_trigger_leaves_too_much_gain():
    """Triggered refresh should not be promoted if it gives back too much unconditional-refresh value."""
    refresh_summary = pd.DataFrame(
        [
            {
                "scenario": "frozen_day_ahead",
                "lock_mae": 100.0,
                "profile_shape_mae": 200.0,
            },
            {
                "scenario": "unconditional_refresh",
                "lock_mae": 70.0,
                "profile_shape_mae": 120.0,
            },
            {
                "scenario": "triggered_refresh",
                "lock_mae": 88.0,
                "profile_shape_mae": 168.0,
            },
        ]
    )
    refresh_decisions = pd.DataFrame({"refresh_triggered": [True, False, True, False]})

    recommendation = module._recommend_day_ahead_refresh(refresh_summary, refresh_decisions)

    assert recommendation["recommended_policy"] == "unconditional_refresh"
    assert bool(recommendation["trigger_rate_in_band"]) is True
    assert bool(recommendation["retains_profile_gain_vs_unconditional"]) is False
    assert bool(recommendation["retains_lock_gain_vs_unconditional"]) is False


def test_select_phase_stack_candidate_can_promote_guard_passing_phase_candidate(monkeypatch):
    """Phase stack selection should choose a guard-passing phase candidate on the held-out scope."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "control_promotion_scope", "held_out_evaluation")
    calibration = pd.DataFrame(
        [
            {
                "candidate_label": "hourly::winner",
                "stack_candidate_policy": "hourly_passthrough",
                "meets_stack_guard": False,
                "lock_mae_p50": 600.0,
                "lock_mae_p90": 650.0,
                "lock_mae": 610.0,
                "profile_shape_mae_p50": 700.0,
                "profile_shape_mae_p90": 710.0,
                "profile_shape_mae": 705.0,
                "minute_path_mae": 720.0,
            },
            {
                "candidate_label": "phase::better",
                "stack_candidate_policy": "phase_candidate",
                "meets_stack_guard": True,
                "lock_mae_p50": 580.0,
                "lock_mae_p90": 600.0,
                "lock_mae": 590.0,
                "profile_shape_mae_p50": 699.0,
                "profile_shape_mae_p90": 708.0,
                "profile_shape_mae": 704.0,
                "minute_path_mae": 715.0,
            },
        ]
    )
    evaluation = pd.DataFrame(
        [
            {
                "candidate_label": "hourly::winner",
                "stack_candidate_policy": "hourly_passthrough",
                "meets_stack_guard": False,
                "lock_mae_p50": 500.0,
                "lock_mae_p90": 540.0,
                "lock_mae": 520.0,
                "profile_shape_mae_p50": 660.0,
                "profile_shape_mae_p90": 680.0,
                "profile_shape_mae": 670.0,
                "minute_path_mae": 690.0,
            },
            {
                "candidate_label": "phase::better",
                "stack_candidate_policy": "phase_candidate",
                "meets_stack_guard": True,
                "lock_mae_p50": 430.0,
                "lock_mae_p90": 470.0,
                "lock_mae": 450.0,
                "profile_shape_mae_p50": 640.0,
                "profile_shape_mae_p90": 665.0,
                "profile_shape_mae": 650.0,
                "minute_path_mae": 680.0,
            },
        ]
    )

    selected, selection_mode = module._select_phase_stack_candidate(
        calibration_benchmark=calibration,
        evaluation_benchmark=evaluation,
        hourly_candidate_label="hourly::winner",
    )

    assert selection_mode == "held_out_phase_stack_candidate_benchmark"
    assert str(selected["candidate_label"]) == "phase::better"


def test_select_phase_stack_candidate_can_fall_back_to_hourly_passthrough(monkeypatch):
    """Phase stack selection should choose hourly passthrough if no phase candidate clears the guard."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "control_promotion_scope", "held_out_evaluation")
    calibration = pd.DataFrame(
        [
            {
                "candidate_label": "hourly::winner",
                "stack_candidate_policy": "hourly_passthrough",
                "meets_stack_guard": False,
                "lock_mae_p50": 600.0,
                "lock_mae_p90": 650.0,
                "lock_mae": 610.0,
                "profile_shape_mae_p50": 700.0,
                "profile_shape_mae_p90": 710.0,
                "profile_shape_mae": 705.0,
                "minute_path_mae": 720.0,
            },
            {
                "candidate_label": "phase::weak",
                "stack_candidate_policy": "phase_candidate",
                "meets_stack_guard": False,
                "lock_mae_p50": 590.0,
                "lock_mae_p90": 640.0,
                "lock_mae": 605.0,
                "profile_shape_mae_p50": 720.0,
                "profile_shape_mae_p90": 740.0,
                "profile_shape_mae": 730.0,
                "minute_path_mae": 725.0,
            },
        ]
    )
    evaluation = pd.DataFrame(
        [
            {
                "candidate_label": "hourly::winner",
                "stack_candidate_policy": "hourly_passthrough",
                "meets_stack_guard": False,
                "lock_mae_p50": 500.0,
                "lock_mae_p90": 540.0,
                "lock_mae": 520.0,
                "profile_shape_mae_p50": 660.0,
                "profile_shape_mae_p90": 680.0,
                "profile_shape_mae": 670.0,
                "minute_path_mae": 690.0,
            },
            {
                "candidate_label": "phase::weak",
                "stack_candidate_policy": "phase_candidate",
                "meets_stack_guard": False,
                "lock_mae_p50": 495.0,
                "lock_mae_p90": 550.0,
                "lock_mae": 510.0,
                "profile_shape_mae_p50": 690.0,
                "profile_shape_mae_p90": 710.0,
                "profile_shape_mae": 700.0,
                "minute_path_mae": 700.0,
            },
        ]
    )

    selected, selection_mode = module._select_phase_stack_candidate(
        calibration_benchmark=calibration,
        evaluation_benchmark=evaluation,
        hourly_candidate_label="hourly::winner",
    )

    assert selection_mode == "held_out_phase_stack_hourly_passthrough"
    assert str(selected["candidate_label"]) == "hourly::winner"


def test_phase_stack_guard_can_fall_back_to_hourly_passthrough(monkeypatch):
    """The stack guard should bypass the phase layer when it barely helps lock MAE and hurts profile shape."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_enabled", True)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_min_lock_gain_pct", 0.01)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_max_profile_degrade_pct", 0.002)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "control_promotion_scope", "held_out_evaluation")
    calibration_summary = pd.DataFrame(
        [
            {"role": "hourly", "lock_mae": 500.0, "profile_shape_mae": 700.0},
            {"role": "phase", "lock_mae": 496.0, "profile_shape_mae": 704.0},
        ]
    )
    evaluation_summary = pd.DataFrame(
        [
            {"role": "hourly", "lock_mae": 300.0, "profile_shape_mae": 400.0},
            {"role": "phase", "lock_mae": 299.5, "profile_shape_mae": 403.0},
        ]
    )

    decision = module._phase_stack_guard_decision(
        calibration_summary=calibration_summary,
        evaluation_summary=evaluation_summary,
        hourly_candidate_label="hourly::winner",
        phase_candidate_label="phase::winner",
    )

    assert decision["recommended_policy"] == "hourly_passthrough"
    assert decision["applied_candidate_label"] == "hourly::winner"
    assert bool(decision["meets_lock_gain_rule"]) is False
    assert bool(decision["meets_profile_rule"]) is False


def test_phase_stack_guard_keeps_phase_when_stack_gain_is_large_enough(monkeypatch):
    """The stack guard should keep the phase layer when it clears both stack-level rules."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_enabled", True)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_min_lock_gain_pct", 0.005)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_max_profile_degrade_pct", 0.01)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "control_promotion_scope", "held_out_evaluation")
    calibration_summary = pd.DataFrame(
        [
            {"role": "hourly", "lock_mae": 500.0, "profile_shape_mae": 700.0},
            {"role": "phase", "lock_mae": 490.0, "profile_shape_mae": 699.0},
        ]
    )
    evaluation_summary = pd.DataFrame(
        [
            {"role": "hourly", "lock_mae": 300.0, "profile_shape_mae": 400.0},
            {"role": "phase", "lock_mae": 296.0, "profile_shape_mae": 401.0},
        ]
    )

    decision = module._phase_stack_guard_decision(
        calibration_summary=calibration_summary,
        evaluation_summary=evaluation_summary,
        hourly_candidate_label="hourly::winner",
        phase_candidate_label="phase::winner",
    )

    assert decision["recommended_policy"] == "phase_candidate"
    assert decision["applied_candidate_label"] == "phase::winner"
    assert bool(decision["meets_lock_gain_rule"]) is True
    assert bool(decision["meets_profile_rule"]) is True


def test_phase_stack_guard_can_fall_back_when_next_lock_regresses(monkeypatch):
    """The phase stack guard should reject a candidate that harms the immediate next-lock surface."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_enabled", True)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_min_lock_gain_pct", 0.0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_max_next_lock_regress_pct", 0.0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_max_profile_degrade_pct", 0.01)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_max_peak_value_regress_pct", 0.05)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_min_peak_hit_gain", 0.0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_max_optimizer_regress_pct", 0.20)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "control_promotion_scope", "held_out_evaluation")
    calibration_summary = pd.DataFrame(
        [
            {
                "role": "hourly",
                "lock_mae": 500.0,
                "next_lock_mae": 140.0,
                "profile_shape_mae": 700.0,
                "peak_value_mae": 800.0,
                "peak_interval_hit_rate": 0.25,
                "optimizer_score": 10.0,
            },
            {
                "role": "phase",
                "lock_mae": 490.0,
                "next_lock_mae": 150.0,
                "profile_shape_mae": 699.0,
                "peak_value_mae": 780.0,
                "peak_interval_hit_rate": 0.30,
                "optimizer_score": 10.0,
            },
        ]
    )
    evaluation_summary = pd.DataFrame(
        [
            {
                "role": "hourly",
                "lock_mae": 300.0,
                "next_lock_mae": 100.0,
                "profile_shape_mae": 400.0,
                "peak_value_mae": 200.0,
                "peak_interval_hit_rate": 0.25,
                "optimizer_score": 10.0,
            },
            {
                "role": "phase",
                "lock_mae": 295.0,
                "next_lock_mae": 105.0,
                "profile_shape_mae": 401.0,
                "peak_value_mae": 180.0,
                "peak_interval_hit_rate": 0.30,
                "optimizer_score": 10.1,
            },
        ]
    )

    decision = module._phase_stack_guard_decision(
        calibration_summary=calibration_summary,
        evaluation_summary=evaluation_summary,
        hourly_candidate_label="hourly::winner",
        phase_candidate_label="phase::winner",
    )

    assert decision["recommended_policy"] == "hourly_passthrough"
    assert bool(decision["meets_lock_gain_rule"]) is True
    assert bool(decision["meets_next_lock_rule"]) is False
    assert bool(decision["meets_peak_value_rule"]) is True
    assert bool(decision["meets_peak_hit_rule"]) is True


def test_phase_stack_guard_can_fall_back_when_optimizer_score_regresses(monkeypatch):
    """The phase stack guard should reject a layer that regresses the optimizer-weighted score too far."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_enabled", True)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_min_lock_gain_pct", 0.0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_max_profile_degrade_pct", 0.01)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_max_optimizer_regress_pct", 0.05)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "control_promotion_scope", "held_out_evaluation")
    calibration_summary = pd.DataFrame(
        [
            {"role": "hourly", "lock_mae": 500.0, "profile_shape_mae": 700.0, "optimizer_score": 10.0},
            {"role": "phase", "lock_mae": 490.0, "profile_shape_mae": 700.0, "optimizer_score": 10.1},
        ]
    )
    evaluation_summary = pd.DataFrame(
        [
            {"role": "hourly", "lock_mae": 300.0, "profile_shape_mae": 400.0, "optimizer_score": 10.0},
            {"role": "phase", "lock_mae": 290.0, "profile_shape_mae": 401.0, "optimizer_score": 10.8},
        ]
    )

    decision = module._phase_stack_guard_decision(
        calibration_summary=calibration_summary,
        evaluation_summary=evaluation_summary,
        hourly_candidate_label="hourly::winner",
        phase_candidate_label="phase::winner",
    )

    assert decision["recommended_policy"] == "hourly_passthrough"
    assert decision["applied_candidate_label"] == "hourly::winner"
    assert bool(decision["meets_lock_gain_rule"]) is True
    assert bool(decision["meets_profile_rule"]) is True
    assert bool(decision["meets_optimizer_rule"]) is False


def test_rolling_phase_stack_guard_can_veto_flat_phase_support(monkeypatch):
    """Broader rolling evidence should demote a phase layer that adds no measurable lock gain."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_require_rolling_support", True)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_rolling_scope", "rolling_evaluation")
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_min_rolling_lock_gain_pct", 0.001)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_max_rolling_next_lock_regress_pct", 0.0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_max_rolling_profile_degrade_pct", 0.0)
    calibration_summary = pd.DataFrame(
        [
            {"role": "hourly", "lock_mae": 500.0, "next_lock_mae": 140.0, "profile_shape_mae": 700.0},
            {"role": "phase", "lock_mae": 495.0, "next_lock_mae": 140.0, "profile_shape_mae": 700.0},
        ]
    )
    evaluation_summary = pd.DataFrame(
        [
            {"role": "hourly", "lock_mae": 300.0, "next_lock_mae": 90.0, "profile_shape_mae": 400.0},
            {"role": "phase", "lock_mae": 300.0, "next_lock_mae": 90.0, "profile_shape_mae": 400.0},
        ]
    )
    combined_summary = pd.DataFrame(
        [
            {"role": "hourly", "lock_mae": 400.0, "next_lock_mae": 115.0, "profile_shape_mae": 550.0},
            {"role": "phase", "lock_mae": 397.5, "next_lock_mae": 115.0, "profile_shape_mae": 550.0},
        ]
    )

    decision = module._rolling_phase_stack_guard_decision(
        calibration_summary=calibration_summary,
        evaluation_summary=evaluation_summary,
        combined_summary=combined_summary,
        hourly_candidate_label="hourly::winner",
        phase_candidate_label="phase::winner",
    )

    assert decision["recommended_policy"] == "hourly_passthrough"
    assert bool(decision["meets_lock_gain_rule"]) is False


def test_rolling_phase_stack_guard_can_veto_peak_regression(monkeypatch):
    """Rolling support should reject a phase layer that regresses peak behavior even when lock gain is positive."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_require_rolling_support", True)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_rolling_scope", "rolling_evaluation")
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_min_rolling_lock_gain_pct", 0.0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_max_rolling_next_lock_regress_pct", 0.0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_max_rolling_profile_degrade_pct", 0.0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_max_rolling_peak_value_regress_pct", 0.0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_min_rolling_peak_hit_gain", 0.0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_guard_max_rolling_optimizer_regress_pct", 0.0)
    calibration_summary = pd.DataFrame(
        [
            {
                "role": "hourly",
                "lock_mae": 500.0,
                "next_lock_mae": 140.0,
                "profile_shape_mae": 700.0,
                "peak_value_mae": 200.0,
                "peak_interval_hit_rate": 0.25,
                "optimizer_score": 8.0,
            },
            {
                "role": "phase",
                "lock_mae": 495.0,
                "next_lock_mae": 140.0,
                "profile_shape_mae": 700.0,
                "peak_value_mae": 240.0,
                "peak_interval_hit_rate": 0.25,
                "optimizer_score": 8.2,
            },
        ]
    )
    decision = module._rolling_phase_stack_guard_decision(
        calibration_summary=calibration_summary,
        evaluation_summary=calibration_summary,
        combined_summary=calibration_summary,
        hourly_candidate_label="hourly::winner",
        phase_candidate_label="phase::winner",
    )

    assert decision["recommended_policy"] == "hourly_passthrough"
    assert bool(decision["meets_peak_value_rule"]) is False


def test_combine_phase_stack_guard_with_rolling_support_applies_veto():
    """The final phase policy should fall back to hourly when the rolling support guard vetoes it."""
    exact_guard = {
        "enabled": True,
        "decision_scope": "held_out_evaluation",
        "recommended_policy": "phase_candidate",
        "applied_candidate_label": "phase::winner",
        "lock_gain_vs_hourly": 5.0,
        "lock_gain_pct_vs_hourly": 0.01,
        "next_lock_regress_vs_hourly": -2.0,
        "next_lock_regress_pct_vs_hourly": 0.0,
        "profile_degrade_vs_hourly": 0.0,
        "profile_degrade_pct_vs_hourly": 0.0,
        "peak_value_regress_vs_hourly": -1.0,
        "peak_value_regress_pct_vs_hourly": 0.0,
        "peak_hit_gain_vs_hourly": 0.1,
        "optimizer_regress_vs_hourly": -0.5,
        "optimizer_regress_pct_vs_hourly": 0.0,
        "meets_lock_gain_rule": True,
        "meets_next_lock_rule": True,
        "meets_profile_rule": True,
        "meets_peak_value_rule": True,
        "meets_peak_hit_rule": True,
        "meets_optimizer_rule": True,
        "reason": "Exact support passed.",
    }
    rolling_support_guard = {
        "enabled": True,
        "required": True,
        "decision_scope": "rolling_evaluation",
        "recommended_policy": "hourly_passthrough",
        "applied_candidate_label": "hourly::winner",
        "lock_gain_vs_hourly": 0.0,
        "lock_gain_pct_vs_hourly": 0.0,
        "next_lock_regress_vs_hourly": 0.0,
        "next_lock_regress_pct_vs_hourly": 0.0,
        "profile_degrade_vs_hourly": 0.0,
        "profile_degrade_pct_vs_hourly": 0.0,
        "peak_value_regress_vs_hourly": float("nan"),
        "peak_value_regress_pct_vs_hourly": float("nan"),
        "peak_hit_gain_vs_hourly": float("nan"),
        "optimizer_regress_vs_hourly": float("nan"),
        "optimizer_regress_pct_vs_hourly": float("nan"),
        "meets_lock_gain_rule": False,
        "meets_next_lock_rule": True,
        "meets_profile_rule": True,
        "meets_peak_value_rule": True,
        "meets_peak_hit_rule": True,
        "meets_optimizer_rule": True,
        "reason": "Rolling support failed.",
    }

    combined = module._combine_phase_stack_guard_with_rolling_support(
        phase_stack_guard=exact_guard,
        rolling_support_guard=rolling_support_guard,
        hourly_candidate_label="hourly::winner",
    )

    assert combined["recommended_policy"] == "hourly_passthrough"
    assert combined["applied_candidate_label"] == "hourly::winner"
    assert bool(combined["rolling_support_applied_veto"]) is True


def test_select_phase_stack_candidate_can_rank_on_optimizer_score(monkeypatch):
    """Phase stack candidate selection should honor the optimizer-aware stack metric when configured."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "control_promotion_scope", "held_out_evaluation")
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_selection_metric", "optimizer_score")
    calibration = pd.DataFrame(
        [
            {
                "candidate_label": "hourly::winner",
                "stack_candidate_policy": "hourly_passthrough",
                "meets_stack_guard": False,
            }
        ]
    )
    evaluation = pd.DataFrame(
        [
            {
                "candidate_label": "hourly::winner",
                "stack_candidate_policy": "hourly_passthrough",
                "meets_stack_guard": False,
                "optimizer_score_p50": 40.0,
                "optimizer_score_p90": 42.0,
                "optimizer_score": 41.0,
                "next_lock_mae_p50": 120.0,
                "peak_interval_miss_rate_p50": 0.50,
                "peak_value_mae_p50": 300.0,
                "lock_mae_p50": 500.0,
                "profile_shape_mae_p50": 650.0,
            },
            {
                "candidate_label": "phase::lock_only",
                "stack_candidate_policy": "phase_candidate",
                "meets_stack_guard": True,
                "optimizer_score_p50": 32.0,
                "optimizer_score_p90": 38.0,
                "optimizer_score": 35.0,
                "next_lock_mae_p50": 125.0,
                "peak_interval_miss_rate_p50": 0.60,
                "peak_value_mae_p50": 330.0,
                "lock_mae_p50": 430.0,
                "profile_shape_mae_p50": 640.0,
            },
            {
                "candidate_label": "phase::optimizer_best",
                "stack_candidate_policy": "phase_candidate",
                "meets_stack_guard": True,
                "optimizer_score_p50": 28.0,
                "optimizer_score_p90": 30.0,
                "optimizer_score": 29.0,
                "next_lock_mae_p50": 90.0,
                "peak_interval_miss_rate_p50": 0.20,
                "peak_value_mae_p50": 180.0,
                "lock_mae_p50": 450.0,
                "profile_shape_mae_p50": 645.0,
            },
        ]
    )

    selected, selection_mode = module._select_phase_stack_candidate(
        calibration_benchmark=calibration,
        evaluation_benchmark=evaluation,
        hourly_candidate_label="hourly::winner",
    )

    assert selection_mode == "held_out_phase_stack_candidate_benchmark"
    assert str(selected["candidate_label"]) == "phase::optimizer_best"


def test_resolve_control_origin_sets_separates_calibration_and_evaluation_splits(monkeypatch):
    """Stage-10 should benchmark on calibration splits and headline on evaluation splits."""
    minute_index = pd.date_range("2026-03-26T00:00:00", periods=6 * 24 * 60, freq="1min")
    base = pd.DataFrame(
        {
            "timestamp": minute_index,
            "day_idx": 26 + ((minute_index - minute_index[0]).days.astype(int)),
        }
    )
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "day_ahead_horizon_minutes", 1440)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "actual_resolution", "1min")
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "cycle_origin_hour", 0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "cycle_origin_minute", 0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "cycle_origin_stride_minutes", 360)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "calibration_splits", ["validate"])
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "evaluation_splits", ["test"])
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "max_cycles", 8)

    catalog, calibration_origins, evaluation_origins = module._resolve_control_origin_sets(base)

    assert not catalog.empty
    assert len(calibration_origins) == 8
    assert len(evaluation_origins) == 8
    assert all(pd.Timestamp(value).day in {26, 27} for value in calibration_origins)
    assert all(pd.Timestamp(value).day in {29, 30} for value in evaluation_origins)


def test_resolve_rolling_control_origin_sets_respects_stride_and_per_split_cap(monkeypatch):
    """The broader rolling benchmark should use its own stride and keep a per-split cap."""
    minute_index = pd.date_range("2026-03-26T00:00:00", periods=6 * 24 * 60, freq="1min")
    base = pd.DataFrame(
        {
            "timestamp": minute_index,
            "day_idx": 26 + ((minute_index - minute_index[0]).days.astype(int)),
        }
    )
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "rolling_benchmark_enabled", True)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "day_ahead_horizon_minutes", 1440)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "actual_resolution", "1min")
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "cycle_origin_hour", 0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "cycle_origin_minute", 0)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "rolling_benchmark_origin_stride_minutes", 180)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "rolling_benchmark_max_cycles", 3)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "calibration_splits", ["validate"])
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "evaluation_splits", ["test"])

    catalog, calibration_origins, evaluation_origins = module._resolve_rolling_control_origin_sets(base)

    assert not catalog.empty
    assert len(calibration_origins) == 3
    assert len(evaluation_origins) == 3
    assert all(pd.Timestamp(value).minute in {0, 180 % 60} for value in calibration_origins)
    assert all(pd.Timestamp(value).day in {26, 27} for value in calibration_origins)
    assert all(pd.Timestamp(value).day in {29, 30} for value in evaluation_origins)


def test_day_ahead_refresh_threshold_candidates_include_defaults_and_quantile_grid(monkeypatch):
    """Threshold search should include the configured baseline plus quantile-derived challengers."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "day_ahead_refresh_threshold_quantiles", [0.5, 0.9])
    monkeypatch.setitem(
        module.MULTIRES_FORECAST_CONTROL,
        "day_ahead_refresh_candidate_trigger_modes",
        ["any", "residual_and_activity"],
    )
    signal_frame = pd.DataFrame(
        {
            "residual_mae_pct": [5.0, 10.0, 20.0, 30.0],
            "transition_state_mismatch": [False, True, False, True],
            "transition_residual_mae_pct": [4.0, 8.0, 16.0, 24.0],
            "activity_ratio_shift": [0.02, 0.05, 0.10, 0.20],
        }
    )

    candidates = module._day_ahead_refresh_threshold_candidates(signal_frame)

    assert candidates[0]["threshold_source"] == "configured_defaults"
    assert candidates[0]["trigger_mode"] == str(module.MULTIRES_FORECAST_CONTROL["day_ahead_refresh_trigger_mode"])
    assert any(candidate["threshold_source"] == "calibration_quantile_grid" for candidate in candidates)
    assert {str(candidate["trigger_mode"]) for candidate in candidates} == {"any", "residual_and_activity"}
    assert len(candidates) > 1


def test_evaluate_day_ahead_refresh_policy_reuses_static_cycle_metrics(monkeypatch):
    """Frozen and unconditional scenario metrics should not be recomputed for every threshold candidate."""

    def _metrics(seed: float) -> dict[str, float]:
        return {
            "minute_path_mae": seed + 1.0,
            "minute_path_mae_pct": seed + 2.0,
            "lock_mae": seed + 3.0,
            "lock_mae_pct": seed + 4.0,
            "profile_shape_mae": seed + 5.0,
            "profile_shape_mae_pct": seed + 6.0,
            "energy_mae": seed + 7.0,
            "energy_mae_pct": seed + 8.0,
        }

    cycle_origin = pd.Timestamp("2026-03-01T00:00:00")
    refresh_origin = cycle_origin + pd.Timedelta(minutes=60)
    minute_index = pd.date_range(cycle_origin + pd.Timedelta(minutes=1), periods=3, freq="1min")
    minute_frame = pd.DataFrame(
        {
            "cycle_origin_timestamp": cycle_origin.isoformat(),
            "timestamp": minute_index,
            "actual_load": [10.0, 12.0, 14.0],
            "day_ahead_pred": [9.0, 11.0, 13.0],
            "unconditional_refresh_pred": [8.0, 10.0, 12.0],
        }
    )
    cached_frozen = _metrics(10.0)
    cached_unconditional = _metrics(20.0)
    triggered_metrics = _metrics(30.0)
    scenario_metric_calls: list[str] = []

    def _fake_apply_thresholds(*, signal_row, thresholds=None):
        return {
            "cycle_origin_timestamp": str(signal_row["cycle_origin_timestamp"]),
            "refresh_origin_timestamp": str(signal_row["refresh_origin_timestamp"]),
            "refresh_triggered": True,
        }

    def _fake_apply_rollout_updates(base, detail_by_origin, candidate_label, update_origins, horizon_minutes):
        return pd.Series([7.0, 7.0, 7.0], index=base.index, dtype=float)

    def _fake_scenario_cycle_metrics(*, minute_frame, prediction_column, lock_interval_minutes):
        scenario_metric_calls.append(str(prediction_column))
        assert prediction_column == "triggered_refresh_pred"
        return dict(triggered_metrics)

    monkeypatch.setattr(module, "_apply_day_ahead_refresh_thresholds", _fake_apply_thresholds)
    monkeypatch.setattr(module, "_apply_rollout_updates", _fake_apply_rollout_updates)
    monkeypatch.setattr(module, "_scenario_cycle_metrics", _fake_scenario_cycle_metrics)

    decisions, refresh_by_cycle, refresh_summary = module._evaluate_day_ahead_refresh_policy(
        cycle_inputs=[
            {
                "cycle_origin_timestamp": cycle_origin,
                "cycle_origin_label": cycle_origin.isoformat(),
                "cycle_refresh_origins": [refresh_origin],
                "minute_frame": minute_frame,
                "frozen_day_ahead_metrics": dict(cached_frozen),
                "unconditional_refresh_metrics": dict(cached_unconditional),
            }
        ],
        signal_frame=pd.DataFrame(
            [
                {
                    "cycle_origin_timestamp": cycle_origin.isoformat(),
                    "refresh_origin_timestamp": refresh_origin.isoformat(),
                }
            ]
        ),
        thresholds={"trigger_mode": "residual_or_transition"},
        day_ahead_refresh={"candidate_label": "refresh_model", "result": {"detail_by_origin": pd.DataFrame()}},
        result_key="result",
        day_ahead_horizon=120,
        lock_interval=15,
    )

    assert scenario_metric_calls == ["triggered_refresh_pred"]
    assert float(refresh_by_cycle.iloc[0]["frozen_day_ahead_lock_mae"]) == cached_frozen["lock_mae"]
    assert (
        float(refresh_by_cycle.iloc[0]["unconditional_refresh_lock_mae"])
        == cached_unconditional["lock_mae"]
    )
    assert float(refresh_by_cycle.iloc[0]["triggered_refresh_lock_mae"]) == triggered_metrics["lock_mae"]
    assert bool(decisions.iloc[0]["refresh_triggered"]) is True
    assert set(refresh_summary["scenario"]) == {
        "frozen_day_ahead",
        "unconditional_refresh",
        "triggered_refresh",
    }


def test_select_day_ahead_refresh_thresholds_reuses_chosen_threshold_outputs(monkeypatch):
    """Threshold search should not rerun the chosen policy after evaluating the calibration grid."""

    candidates = [
        {
            "residual_drift_mae_pct_threshold": 1.0,
            "transition_mae_pct_threshold": 1.0,
            "activity_ratio_shift_threshold": 0.1,
            "trigger_mode": "residual_or_transition",
        },
        {
            "residual_drift_mae_pct_threshold": 2.0,
            "transition_mae_pct_threshold": 2.0,
            "activity_ratio_shift_threshold": 0.2,
            "trigger_mode": "residual_or_transition",
        },
    ]
    evaluate_calls: list[float] = []
    decision_calls: list[float] = []

    def _fake_threshold_candidates(signal_frame):
        return list(candidates)

    def _fake_decisions(signal_frame, *, thresholds):
        threshold_value = float(thresholds["residual_drift_mae_pct_threshold"])
        decision_calls.append(threshold_value)
        return pd.DataFrame(
            {
                "cycle_origin_timestamp": ["cycle-origin"],
                "refresh_origin_timestamp": [f"refresh-{int(threshold_value)}"],
                "refresh_triggered": [threshold_value == 1.0],
            }
        )

    def _fake_evaluate_from_decisions(
        *,
        cycle_inputs,
        decision_rows,
        day_ahead_refresh,
        result_key,
        day_ahead_horizon,
        lock_interval,
    ):
        threshold_value = 1.0 if bool(decision_rows.iloc[0]["refresh_triggered"]) else 2.0
        evaluate_calls.append(threshold_value)
        refresh_by_cycle = pd.DataFrame({"cycle_origin_timestamp": [f"cycle-{int(threshold_value)}"]})
        refresh_summary = pd.DataFrame(
            [
                {"scenario": "frozen_day_ahead", "lock_mae": 10.0, "profile_shape_mae": 10.0},
                {"scenario": "unconditional_refresh", "lock_mae": 8.0, "profile_shape_mae": 8.0},
                {
                    "scenario": "triggered_refresh",
                    "lock_mae": 7.0 if threshold_value == 1.0 else 9.5,
                    "profile_shape_mae": 7.0 if threshold_value == 1.0 else 9.5,
                },
            ]
        )
        return refresh_by_cycle, refresh_summary

    def _fake_recommend(refresh_summary, refresh_decisions):
        triggered = refresh_summary.loc[refresh_summary["scenario"].eq("triggered_refresh")].iloc[0]
        return {
            "recommended_policy": "triggered_refresh" if float(triggered["lock_mae"]) < 8.0 else "frozen_day_ahead",
            "trigger_rate": 0.5,
            "trigger_rate_in_band": True,
            "retains_profile_gain_vs_unconditional": True,
            "retains_lock_gain_vs_unconditional": True,
            "triggered_profile_gain_fraction_vs_unconditional": 1.0,
            "triggered_lock_gain_fraction_vs_unconditional": 1.0,
            "reason": "ok",
        }

    monkeypatch.setattr(module, "_day_ahead_refresh_threshold_candidates", _fake_threshold_candidates)
    monkeypatch.setattr(module, "_day_ahead_refresh_decisions", _fake_decisions)
    monkeypatch.setattr(module, "_evaluate_day_ahead_refresh_policy_from_decisions", _fake_evaluate_from_decisions)
    monkeypatch.setattr(module, "_recommend_day_ahead_refresh", _fake_recommend)

    chosen_thresholds, grid, chosen_decisions, chosen_refresh_by_cycle, chosen_refresh_summary = (
        module._select_day_ahead_refresh_thresholds(
            calibration_cycle_inputs=[{"cycle_origin_timestamp": pd.Timestamp("2026-03-01T00:00:00")}],
            calibration_signal_frame=pd.DataFrame([{"signal": 1.0}]),
            day_ahead_refresh={"candidate_label": "refresh_model"},
            result_key="benchmark_result",
            day_ahead_horizon=1440,
            lock_interval=15,
        )
    )

    assert decision_calls == [1.0, 2.0]
    assert evaluate_calls == [1.0, 2.0]
    assert float(chosen_thresholds["residual_drift_mae_pct_threshold"]) == 1.0
    assert len(grid) == 2
    assert bool(chosen_decisions.iloc[0]["refresh_triggered"]) is True
    assert str(chosen_refresh_by_cycle.iloc[0]["cycle_origin_timestamp"]) == "cycle-1"
    assert float(
        chosen_refresh_summary.loc[
            chosen_refresh_summary["scenario"].eq("triggered_refresh"),
            "lock_mae",
        ].iloc[0]
    ) == 7.0


def test_select_day_ahead_refresh_thresholds_dedupes_decision_equivalent_candidates(monkeypatch):
    """Threshold candidates with identical trigger masks should share one expensive replay evaluation."""

    candidates = [
        {
            "residual_drift_mae_pct_threshold": 1.0,
            "transition_mae_pct_threshold": 1.0,
            "activity_ratio_shift_threshold": 0.1,
            "trigger_mode": "residual_or_transition",
        },
        {
            "residual_drift_mae_pct_threshold": 1.5,
            "transition_mae_pct_threshold": 1.5,
            "activity_ratio_shift_threshold": 0.15,
            "trigger_mode": "residual_or_transition",
        },
    ]
    evaluate_calls = 0

    def _fake_threshold_candidates(signal_frame):
        return list(candidates)

    def _fake_decisions(signal_frame, *, thresholds):
        return pd.DataFrame(
            {
                "cycle_origin_timestamp": ["cycle-origin"],
                "refresh_origin_timestamp": ["refresh-origin"],
                "refresh_triggered": [True],
            }
        )

    def _fake_evaluate_from_decisions(
        *,
        cycle_inputs,
        decision_rows,
        day_ahead_refresh,
        result_key,
        day_ahead_horizon,
        lock_interval,
    ):
        nonlocal evaluate_calls
        evaluate_calls += 1
        refresh_by_cycle = pd.DataFrame({"cycle_origin_timestamp": ["cycle-1"]})
        refresh_summary = pd.DataFrame(
            [
                {"scenario": "frozen_day_ahead", "lock_mae": 10.0, "profile_shape_mae": 10.0},
                {"scenario": "unconditional_refresh", "lock_mae": 8.0, "profile_shape_mae": 8.0},
                {"scenario": "triggered_refresh", "lock_mae": 7.0, "profile_shape_mae": 7.0},
            ]
        )
        return refresh_by_cycle, refresh_summary

    def _fake_recommend(refresh_summary, refresh_decisions):
        return {
            "recommended_policy": "triggered_refresh",
            "trigger_rate": 0.5,
            "trigger_rate_in_band": True,
            "retains_profile_gain_vs_unconditional": True,
            "retains_lock_gain_vs_unconditional": True,
            "triggered_profile_gain_fraction_vs_unconditional": 1.0,
            "triggered_lock_gain_fraction_vs_unconditional": 1.0,
            "reason": "ok",
        }

    monkeypatch.setattr(module, "_day_ahead_refresh_threshold_candidates", _fake_threshold_candidates)
    monkeypatch.setattr(module, "_day_ahead_refresh_decisions", _fake_decisions)
    monkeypatch.setattr(module, "_evaluate_day_ahead_refresh_policy_from_decisions", _fake_evaluate_from_decisions)
    monkeypatch.setattr(module, "_recommend_day_ahead_refresh", _fake_recommend)

    _, grid, _, _, _ = module._select_day_ahead_refresh_thresholds(
        calibration_cycle_inputs=[{"cycle_origin_timestamp": pd.Timestamp("2026-03-01T00:00:00")}],
        calibration_signal_frame=pd.DataFrame([{"signal": 1.0}]),
        day_ahead_refresh={"candidate_label": "refresh_model"},
        result_key="benchmark_result",
        day_ahead_horizon=1440,
        lock_interval=15,
    )

    assert evaluate_calls == 1
    assert int(grid["decision_signature_duplicate"].astype(bool).sum()) == 1
    representative = grid.loc[grid["decision_signature_duplicate"].astype(bool).eq(False)].iloc[0]
    duplicate = grid.loc[grid["decision_signature_duplicate"].astype(bool)].iloc[0]
    assert int(representative["decision_signature_origin_rank"]) == 1
    assert int(duplicate["decision_signature_origin_rank"]) == 1


def test_apply_day_ahead_refresh_thresholds_requires_real_transition_mismatch():
    """Transition-triggered refresh should not fire on transitions unless the activity state actually disagrees."""
    signal_row = {
        "residual_mae_pct": 40.0,
        "workday_transition_active": True,
        "transition_state_mismatch": False,
        "transition_residual_mae_pct": 40.0,
        "activity_ratio_shift": 0.01,
    }

    decision = module._apply_day_ahead_refresh_thresholds(
        signal_row=signal_row,
        thresholds={
            "residual_drift_mae_pct_threshold": 50.0,
            "transition_mae_pct_threshold": 10.0,
            "activity_ratio_shift_threshold": 0.2,
        },
    )

    assert bool(decision["transition_mismatch_trigger"]) is False
    assert bool(decision["refresh_triggered"]) is False


def test_apply_day_ahead_refresh_thresholds_can_require_residual_and_activity():
    """Composite trigger modes should only fire when all required signals are active."""
    residual_only = module._apply_day_ahead_refresh_thresholds(
        signal_row={
            "residual_mae_pct": 25.0,
            "workday_transition_active": False,
            "transition_state_mismatch": False,
            "transition_residual_mae_pct": 0.0,
            "activity_ratio_shift": 0.01,
        },
        thresholds={
            "residual_drift_mae_pct_threshold": 20.0,
            "transition_mae_pct_threshold": 10.0,
            "activity_ratio_shift_threshold": 0.05,
            "trigger_mode": "residual_and_activity",
        },
    )
    both_active = module._apply_day_ahead_refresh_thresholds(
        signal_row={
            "residual_mae_pct": 25.0,
            "workday_transition_active": False,
            "transition_state_mismatch": False,
            "transition_residual_mae_pct": 0.0,
            "activity_ratio_shift": 0.10,
        },
        thresholds={
            "residual_drift_mae_pct_threshold": 20.0,
            "transition_mae_pct_threshold": 10.0,
            "activity_ratio_shift_threshold": 0.05,
            "trigger_mode": "residual_and_activity",
        },
    )

    assert bool(residual_only["residual_drift_trigger"]) is True
    assert bool(residual_only["activity_profile_shift_trigger"]) is False
    assert bool(residual_only["refresh_triggered"]) is False
    assert bool(both_active["refresh_triggered"]) is True
    assert str(both_active["trigger_mode"]) == "residual_and_activity"


def test_apply_day_ahead_refresh_thresholds_can_gate_activity_to_active_band():
    """Activity-only trigger modes should suppress low-activity overnight drift."""
    inactive_decision = module._apply_day_ahead_refresh_thresholds(
        signal_row={
            "residual_mae_pct": 5.0,
            "workday_transition_active": False,
            "transition_state_mismatch": False,
            "transition_residual_mae_pct": 0.0,
            "activity_ratio_shift": 0.20,
            "expected_active_flag": 0.0,
            "actual_active_flag": 0.0,
        },
        thresholds={
            "residual_drift_mae_pct_threshold": 20.0,
            "transition_mae_pct_threshold": 10.0,
            "activity_ratio_shift_threshold": 0.05,
            "trigger_mode": "activity_active_band",
        },
    )
    active_decision = module._apply_day_ahead_refresh_thresholds(
        signal_row={
            "residual_mae_pct": 5.0,
            "workday_transition_active": False,
            "transition_state_mismatch": False,
            "transition_residual_mae_pct": 0.0,
            "activity_ratio_shift": 0.20,
            "expected_active_flag": 1.0,
            "actual_active_flag": 1.0,
        },
        thresholds={
            "residual_drift_mae_pct_threshold": 20.0,
            "transition_mae_pct_threshold": 10.0,
            "activity_ratio_shift_threshold": 0.05,
            "trigger_mode": "activity_active_band",
        },
    )

    assert bool(inactive_decision["activity_profile_shift_trigger"]) is True
    assert bool(inactive_decision["refresh_triggered"]) is False
    assert bool(active_decision["refresh_triggered"]) is True


def test_score_nowcast_candidate_predictions_supports_optimizer_score_selection():
    """Direct nowcast scoring should expose optimizer_score without requiring a synthetic cycle-frame column."""
    origin = pd.Timestamp("2026-03-01T00:00:00")
    minute_index = pd.date_range(origin + pd.Timedelta(minutes=1), periods=15, freq="1min")
    minute_timeline = pd.DataFrame(
        {
            "cycle_origin_timestamp": [origin.isoformat()] * len(minute_index),
            "timestamp": minute_index,
            "actual_load": [10.0] * 10 + [30.0] * 5,
            "phase_pred": [9.0] * len(minute_index),
        }
    )
    prediction_series = pd.Series([10.0] * 10 + [28.0] * 5, index=minute_index, dtype=float)

    scored = module._score_nowcast_candidate_predictions(
        minute_timeline=minute_timeline,
        prediction_series=prediction_series,
        candidate_label="candidate_x",
        candidate_type="learned",
        source_model_label="model_x",
        target_mode="raw",
        selection_metric="optimizer_score",
    )

    assert scored["selection_metric_name"] == "optimizer_score"
    assert pd.notna(scored["optimizer_score"])
    assert scored["selection_metric_value"] == scored["optimizer_score"]


def test_calibrate_nowcast_control_blend_can_pick_intermediate_weight(monkeypatch):
    """Minute-layer control calibration should be able to pick a persistence-blend weight."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "nowcast_control_blend_enabled", True)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "nowcast_control_blend_weights", [0.2, 0.4, 0.8])

    def _fake_score(**kwargs):
        label = str(kwargs["candidate_label"])
        weight = 1.0
        if "|control_blend_w" in label:
            weight = float(label.rsplit("w", 1)[1])
        return {
            "candidate_label": label,
            "candidate_type": kwargs["candidate_type"],
            "source_model_label": kwargs["source_model_label"],
            "target_mode": kwargs["target_mode"],
            "minute_path_mae": abs(weight - 0.4) + 1.0,
            "minute_path_mae_pct": abs(weight - 0.4) + 1.0,
            "lock_mae": abs(weight - 0.4),
            "lock_mae_pct": abs(weight - 0.4),
            "profile_shape_mae": abs(weight - 0.4) + 2.0,
            "profile_shape_mae_pct": abs(weight - 0.4) + 2.0,
            "energy_mae": abs(weight - 0.4) + 3.0,
            "energy_mae_pct": abs(weight - 0.4) + 3.0,
            "mean_coverage": 1.0,
            "origin_n": 2,
            "control_layer": "nowcast",
            "selection_metric_name": kwargs["selection_metric"],
            "selection_metric_value": abs(weight - 0.4),
            "selection_metric_pct": abs(weight - 0.4),
        }

    monkeypatch.setattr(module, "_score_nowcast_candidate_predictions", _fake_score)

    calibrated = module._calibrate_nowcast_control_blend(
        candidate={
            "candidate_label": "minimal_phase/hgb-balanced/raw+blend",
            "candidate_type": "learned",
            "model_label": "hgb-balanced",
            "target_mode": "raw+blend",
        },
        benchmark_minute_timeline=pd.DataFrame(),
        evaluation_minute_timeline=pd.DataFrame(),
        benchmark_candidate_series=pd.Series([2.0, 4.0], index=pd.date_range("2026-03-01", periods=2, freq="1min")),
        evaluation_candidate_series=pd.Series([2.0, 4.0], index=pd.date_range("2026-03-02", periods=2, freq="1min")),
        benchmark_persistence_series=pd.Series([1.0, 1.0], index=pd.date_range("2026-03-01", periods=2, freq="1min")),
        evaluation_persistence_series=pd.Series([1.0, 1.0], index=pd.date_range("2026-03-02", periods=2, freq="1min")),
        selection_metric="lock_mae",
    )

    assert calibrated is not None
    assert str(calibrated["candidate_label"]).endswith("|control_blend_w0.40")
    assert float(calibrated["control_blend_weight"]) == 0.4


def test_benchmark_nowcast_layer_can_promote_held_out_evaluation_winner(monkeypatch):
    """The exact-control minute selector should promote the held-out winner when configured to do so."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "optimize_replayed_candidates", True)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "control_promotion_scope", "held_out_evaluation")
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "nowcast_control_blend_enabled", False)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "nowcast_control_bucket_blend_enabled", False)

    candidate_pool = [
        {
            "candidate_label": "persistence",
            "candidate_type": "baseline",
            "resolution": "1min",
            "feature_set": "baseline",
            "model_label": "persistence",
            "target_mode": "baseline",
            "artifact_path": "outputs/005_performance/demo",
            "pool_source_type": "upstream",
            "pool_source_run_id": "demo",
        },
        {
            "candidate_label": "minimal_phase/hgb-balanced/raw+blend",
            "candidate_type": "learned",
            "resolution": "1min",
            "feature_set": "minimal_phase",
            "model_label": "hgb-balanced",
            "target_mode": "raw+blend",
            "artifact_path": "outputs/005_performance/demo",
            "pool_source_type": "upstream",
            "pool_source_run_id": "demo",
        },
    ]
    minute_index = pd.date_range("2026-03-01T00:01:00", periods=4, freq="1min")

    def _fake_predictions(*, candidate, context):
        value = 10.0 if str(candidate["candidate_label"]) == "persistence" else 11.0
        timestamps = minute_index if str(context["scope"]) == "calibration" else minute_index + pd.Timedelta(days=1)
        return pd.DataFrame({"timestamp": timestamps, "predicted_load": [value] * len(timestamps)})

    def _fake_score(*, minute_timeline, candidate_label, candidate_type, source_model_label, target_mode, selection_metric, prediction_series):
        scope = str(minute_timeline["scope"].iloc[0])
        is_learned = str(candidate_label) != "persistence"
        if scope == "calibration":
            metric = 1.1 if is_learned else 1.0
        else:
            metric = 0.5 if is_learned else 2.0
        return {
            "candidate_label": str(candidate_label),
            "candidate_type": str(candidate_type),
            "source_model_label": str(source_model_label),
            "target_mode": str(target_mode),
            "minute_path_mae": float(metric + 1.0),
            "minute_path_mae_pct": float(metric + 1.0),
            "lock_mae": float(metric),
            "lock_mae_pct": float(metric),
            "profile_shape_mae": float(metric + 2.0),
            "profile_shape_mae_pct": float(metric + 2.0),
            "energy_mae": float(metric + 3.0),
            "energy_mae_pct": float(metric + 3.0),
            "mean_coverage": 1.0,
            "origin_n": 1,
            "control_layer": "nowcast",
            "selection_metric_name": str(selection_metric),
            "selection_metric_value": float(metric),
            "selection_metric_pct": float(metric),
        }

    monkeypatch.setattr(module, "_load_stage5_nowcast_anchor", lambda: candidate_pool[0])
    monkeypatch.setattr(module, "_load_stage5_nowcast_candidate_pool", lambda upstream_anchor: candidate_pool)
    monkeypatch.setattr(
        module,
        "_stage5_nowcast_contexts",
        lambda: {"calibration": {"scope": "calibration"}, "evaluation": {"scope": "evaluation"}},
    )
    monkeypatch.setattr(module, "_stage5_candidate_predictions", _fake_predictions)
    monkeypatch.setattr(module, "_score_nowcast_candidate_predictions", _fake_score)

    benchmark_minute_timeline = pd.DataFrame(
        {
            "timestamp": minute_index,
            "actual_load": [10.0, 10.0, 10.0, 10.0],
            "cycle_origin_timestamp": ["2026-03-01T00:00:00"] * len(minute_index),
            "scope": ["calibration"] * len(minute_index),
        }
    )
    evaluation_minute_timeline = pd.DataFrame(
        {
            "timestamp": minute_index + pd.Timedelta(days=1),
            "actual_load": [10.0, 10.0, 10.0, 10.0],
            "cycle_origin_timestamp": ["2026-03-02T00:00:00"] * len(minute_index),
            "scope": ["evaluation"] * len(minute_index),
        }
    )

    selected = module._benchmark_nowcast_layer(
        benchmark_minute_timeline=benchmark_minute_timeline,
        evaluation_minute_timeline=evaluation_minute_timeline,
    )

    assert str(selected["candidate_label"]) == "minimal_phase/hgb-balanced/raw+blend"
    assert str(selected["control_selection_mode"]) == "held_out_control_layer_candidate_benchmark"


def test_selected_nowcast_prediction_series_can_replay_bucket_blend(monkeypatch):
    """Persisted nowcast metadata should be sufficient to reconstruct a bucketed Stage-5 blend."""
    minute_index = pd.date_range("2026-03-02T00:00:00", periods=15, freq="1min")

    def _fake_predictions(*, candidate, context):
        if str(candidate["candidate_label"]) == "persistence":
            values = [10.0] * len(minute_index)
        else:
            values = [20.0] * 5 + [40.0] * 5 + [60.0] * 5
        return pd.DataFrame({"timestamp": minute_index, "predicted_load": values})

    candidate_pool = [
        {
            "candidate_label": "persistence",
            "candidate_type": "baseline",
            "resolution": "1min",
            "feature_set": "baseline",
            "model_label": "persistence",
            "target_mode": "baseline",
        },
        {
            "candidate_label": "curated_ramp/hgb-balanced/residual",
            "candidate_type": "learned",
            "resolution": "1min",
            "feature_set": "curated_ramp",
            "model_label": "hgb-balanced",
            "target_mode": "residual",
        },
    ]
    monkeypatch.setattr(module, "_stage5_candidate_predictions", _fake_predictions)
    monkeypatch.setattr(module, "_load_stage5_nowcast_candidate_pool", lambda upstream_anchor: candidate_pool)
    monkeypatch.setattr(
        module,
        "_stage5_nowcast_contexts",
        lambda: {"calibration": {"scope": "calibration"}, "evaluation": {"scope": "evaluation"}},
    )

    series = module._selected_nowcast_prediction_series(
        nowcast_anchor={
            "candidate_label": "curated_ramp/hgb-balanced/residual|control_bucket_blend_b5",
            "blend_base_candidate_label": "curated_ramp/hgb-balanced/residual",
            "control_bucket_size_minutes": 5,
            "control_bucket_weights_json": '{"0": 0.05, "5": 0.0, "10": 0.0}',
            "upstream_anchor": candidate_pool[0],
        },
        scope_name="evaluation",
    )

    assert float(series.iloc[0]) == 10.5
    assert float(series.iloc[4]) == 10.5
    assert float(series.iloc[5]) == 10.0
    assert float(series.iloc[-1]) == 10.0


def test_phase_bucket_policy_from_origin_metrics_can_route_by_quarter_hour():
    """Origin-level phase metrics should produce a deterministic bucket routing policy."""
    detail = pd.DataFrame(
        [
            {"origin_timestamp": "2026-03-01T00:00:00", "candidate_label": "candidate_a", "next_lock_mae": 2.0},
            {"origin_timestamp": "2026-03-01T00:00:00", "candidate_label": "candidate_b", "next_lock_mae": 4.0},
            {"origin_timestamp": "2026-03-01T00:15:00", "candidate_label": "candidate_a", "next_lock_mae": 5.0},
            {"origin_timestamp": "2026-03-01T00:15:00", "candidate_label": "candidate_b", "next_lock_mae": 1.0},
            {"origin_timestamp": "2026-03-01T00:30:00", "candidate_label": "candidate_a", "next_lock_mae": 1.5},
            {"origin_timestamp": "2026-03-01T00:30:00", "candidate_label": "candidate_b", "next_lock_mae": 3.0},
            {"origin_timestamp": "2026-03-01T00:45:00", "candidate_label": "candidate_a", "next_lock_mae": 4.0},
            {"origin_timestamp": "2026-03-01T00:45:00", "candidate_label": "candidate_b", "next_lock_mae": 1.0},
        ]
    )

    policy = module._phase_bucket_policy_from_origin_metrics(
        detail,
        selection_metric="next_lock_mae",
        bucket_minutes=15,
    )

    assert policy == {0: "candidate_a", 15: "candidate_b", 30: "candidate_a", 45: "candidate_b"}


def test_selected_phase_series_for_scope_can_replay_bucket_portfolio():
    """Phase bucket portfolios should route each update origin through its mapped candidate label."""
    timestamps = pd.date_range("2026-03-01T00:01:00", periods=30, freq="1min")
    minute_timeline = pd.DataFrame({"timestamp": timestamps, "hourly_pred": [10.0] * len(timestamps)})
    phase_detail = pd.DataFrame(
        [
            {
                "origin_timestamp": "2026-03-01T00:00:00",
                "candidate_label": "candidate_a",
                "forecast_timestamp": timestamp.isoformat(),
                "predicted_load": 20.0,
            }
            for timestamp in timestamps[:15]
        ]
        + [
            {
                "origin_timestamp": "2026-03-01T00:15:00",
                "candidate_label": "candidate_b",
                "forecast_timestamp": timestamp.isoformat(),
                "predicted_load": 30.0,
            }
            for timestamp in timestamps[15:30]
        ]
    )

    series = module._selected_phase_series_for_scope(
        minute_timeline=minute_timeline,
        hourly_pred_column="hourly_pred",
        phase_replay_metadata={
            "mode": "phase_bucket_portfolio",
            "bucket_policy_json": '{"0": "candidate_a", "15": "candidate_b"}',
            "bucket_granularity_minutes": 15,
        },
        phase_detail_by_origin=phase_detail,
        phase_origins=[pd.Timestamp("2026-03-01T00:00:00"), pd.Timestamp("2026-03-01T00:15:00")],
        phase_horizon=15,
    )

    assert float(series.iloc[0]) == 20.0
    assert float(series.iloc[14]) == 20.0
    assert float(series.iloc[15]) == 30.0
    assert float(series.iloc[-1]) == 30.0


def test_selected_phase_series_for_scope_bucket_portfolio_falls_back_without_detail():
    """Bucketed phase replay should safely fall back to the hourly path when detailed rows are unavailable."""
    timestamps = pd.date_range("2026-03-01T00:01:00", periods=15, freq="1min")
    minute_timeline = pd.DataFrame({"timestamp": timestamps, "hourly_pred": [10.0] * len(timestamps)})

    series = module._selected_phase_series_for_scope(
        minute_timeline=minute_timeline,
        hourly_pred_column="hourly_pred",
        phase_replay_metadata={
            "mode": "phase_bucket_portfolio",
            "bucket_policy_json": '{"0": "candidate_a"}',
            "bucket_granularity_minutes": 15,
        },
        phase_detail_by_origin=pd.DataFrame(),
        phase_origins=[pd.Timestamp("2026-03-01T00:00:00")],
        phase_horizon=15,
    )

    assert series.equals(pd.Series([10.0] * len(timestamps), index=timestamps, dtype=float))


def test_selected_phase_series_for_scope_can_replay_baseline_control_blends():
    """Phase stack replay should reconstruct both global and bucketed baseline-control blends."""
    timestamps = pd.date_range("2026-03-01T00:01:00", periods=15, freq="1min")
    minute_timeline = pd.DataFrame({"timestamp": timestamps, "hourly_pred": [5.0] * len(timestamps)})
    phase_detail = pd.DataFrame(
        [
            {
                "origin_timestamp": "2026-03-01T00:00:00",
                "candidate_label": "candidate",
                "forecast_timestamp": timestamp.isoformat(),
                "predicted_load": 25.0,
            }
            for timestamp in timestamps
        ]
        + [
            {
                "origin_timestamp": "2026-03-01T00:00:00",
                "candidate_label": "baseline",
                "forecast_timestamp": timestamp.isoformat(),
                "predicted_load": 15.0 if timestamp.minute <= 5 else 10.0,
            }
            for timestamp in timestamps
        ]
    )

    blended_series = module._selected_phase_series_for_scope(
        minute_timeline=minute_timeline,
        hourly_pred_column="hourly_pred",
        phase_replay_metadata={
            "mode": "phase_baseline_control_blend",
            "blend_parent_candidate_label": "candidate",
            "reference_candidate_label": "baseline",
            "blend_weight": 0.5,
        },
        phase_detail_by_origin=phase_detail,
        phase_origins=[pd.Timestamp("2026-03-01T00:00:00")],
        phase_horizon=15,
    )
    bucketed_series = module._selected_phase_series_for_scope(
        minute_timeline=minute_timeline,
        hourly_pred_column="hourly_pred",
        phase_replay_metadata={
            "mode": "phase_baseline_bucket_control_blend",
            "blend_parent_candidate_label": "candidate",
            "reference_candidate_label": "baseline",
            "bucket_weight_json": '{"0": 0.0, "5": 1.0, "10": 0.5}',
            "bucket_granularity_minutes": 5,
        },
        phase_detail_by_origin=phase_detail,
        phase_origins=[pd.Timestamp("2026-03-01T00:00:00")],
        phase_horizon=15,
    )

    assert float(blended_series.iloc[0]) == 20.0
    assert float(bucketed_series.iloc[0]) == 15.0
    assert float(bucketed_series.iloc[5]) == 25.0
    assert float(bucketed_series.iloc[10]) == 17.5


def test_resolve_phase_stack_replay_metadata_uses_parent_selection_for_bucket_control_blends():
    """Synthetic phase-stack bucket blends should replay through their parent candidate selection."""
    phase_payload = {
        "candidate_label": "1min/minimal/hgb-balanced::raw",
        "selection": {
            "resolution": "1min",
            "feature_set": "minimal",
            "model_label": "hgb-balanced",
            "requested_candidate_label": "hgb-balanced::raw",
        },
        "candidate_selection_by_pool_rank": {
            3: {
                "resolution": "5min",
                "feature_set": "minimal",
                "model_label": "hgb-frontier-lr010-leaf100",
            }
        },
    }
    selected_row = pd.Series(
        {
            "candidate_label": "5min/minimal/hgb-frontier-lr010-leaf100::anchored_workday_residual|baseline_control_bucket_blend_b5",
            "stack_candidate_family": "phase_baseline_bucket_control_blend",
            "stack_blend_parent_candidate_label": "5min/minimal/hgb-frontier-lr010-leaf100::anchored_workday_residual",
            "stack_reference_candidate_label": "5min/minimal/anchored_workday",
            "stack_bucket_weight_json": json.dumps({0: 0.25, 5: 0.75}),
            "stack_bucket_granularity_minutes": 5.0,
            "replay_pool_rank": 3.0,
            "replay_run_dir": "outputs/007_rollout/commercial_facility/20260319T040700234380Z",
        }
    )

    metadata = module._resolve_phase_stack_replay_metadata(
        phase_payload=phase_payload,
        phase_stack_selected_row=selected_row,
        phase_stack_guard={
            "recommended_policy": "phase_candidate",
            "applied_candidate_label": str(selected_row["candidate_label"]),
        },
    )

    assert metadata["mode"] == "phase_baseline_bucket_control_blend"
    assert metadata["blend_parent_candidate_label"] == (
        "5min/minimal/hgb-frontier-lr010-leaf100::anchored_workday_residual"
    )
    assert metadata["reference_candidate_label"] == "5min/minimal/anchored_workday"
    assert metadata["selection"]["requested_candidate_label"] == "hgb-frontier-lr010-leaf100::anchored_workday_residual"


def test_qualify_phase_replay_detail_allows_reconstructed_bucket_blends_to_match_replayed_labels():
    """Replayed rolling phase detail should be re-qualified before reconstructing stack-level blends."""
    timestamps = pd.date_range("2026-03-01T00:01:00", periods=15, freq="1min")
    minute_timeline = pd.DataFrame({"timestamp": timestamps, "hourly_pred": [5.0] * len(timestamps)})
    raw_detail = pd.DataFrame(
        [
            {
                "origin_timestamp": "2026-03-01T00:00:00",
                "candidate_label": "hgb-frontier-lr010-leaf100::anchored_workday_residual",
                "forecast_timestamp": timestamp.isoformat(),
                "predicted_load": 25.0,
            }
            for timestamp in timestamps
        ]
        + [
            {
                "origin_timestamp": "2026-03-01T00:00:00",
                "candidate_label": "anchored_workday",
                "forecast_timestamp": timestamp.isoformat(),
                "predicted_load": 15.0 if timestamp.minute <= 5 else 10.0,
            }
            for timestamp in timestamps
        ]
    )

    qualified_detail = module._qualify_phase_replay_detail(
        raw_detail,
        replay_selection={
            "resolution": "5min",
            "feature_set": "minimal",
            "model_label": "hgb-frontier-lr010-leaf100",
        },
    )
    series = module._selected_phase_series_for_scope(
        minute_timeline=minute_timeline,
        hourly_pred_column="hourly_pred",
        phase_replay_metadata={
            "mode": "phase_baseline_bucket_control_blend",
            "blend_parent_candidate_label": "5min/minimal/hgb-frontier-lr010-leaf100::anchored_workday_residual",
            "reference_candidate_label": "5min/minimal/anchored_workday",
            "bucket_weight_json": '{"0": 0.0, "5": 1.0, "10": 0.5}',
            "bucket_granularity_minutes": 5,
        },
        phase_detail_by_origin=qualified_detail,
        phase_origins=[pd.Timestamp("2026-03-01T00:00:00")],
        phase_horizon=15,
    )

    assert list(qualified_detail["native_candidate_label"].astype("string").unique()) == [
        "hgb-frontier-lr010-leaf100::anchored_workday_residual",
        "anchored_workday",
    ]
    assert float(series.iloc[0]) == 15.0
    assert float(series.iloc[5]) == 25.0
    assert float(series.iloc[10]) == 17.5


def test_replay_phase_detail_for_scope_uses_selected_only_for_native_selection(monkeypatch, tmp_path: Path):
    """Native phase replay should reuse the stored selection instead of widening to the full family."""
    captured: dict[str, object] = {}

    def fake_run_cached_rollout_evaluation(**kwargs):
        captured["selection"] = dict(kwargs["selection"])
        captured["candidate_scope"] = str(kwargs["candidate_scope"])
        return {
            "detail_by_origin": pd.DataFrame(
                [{"candidate_label": "candidate_a", "predicted_load": 1.0}]
            )
        }

    monkeypatch.setattr(
        module,
        "_representable_selection_origins",
        lambda **kwargs: [pd.Timestamp(value) for value in kwargs["origin_timestamps"]],
    )
    monkeypatch.setattr(module, "_run_cached_rollout_evaluation", fake_run_cached_rollout_evaluation)
    phase_payload = {
        "selection": {
            "resolution": "5min",
            "feature_set": "full",
            "model_label": "upstream_model",
            "requested_candidate_label": "candidate_b",
        },
        "origin_policy": "phase_balanced",
        "selection_target": "next_lock_mae",
    }
    replay_selection = {
        "resolution": "5min",
        "feature_set": "minimal_phase",
        "model_label": "selected_model",
        "requested_candidate_label": "candidate_a",
    }

    detail = module._replay_phase_detail_for_scope(
        phase_payload=phase_payload,
        phase_replay_metadata={"mode": "native_phase_candidate", "selection": replay_selection},
        cache_root=None,
        temp_output_root=tmp_path / "selected",
        origin_timestamps=[pd.Timestamp("2026-03-01T00:00:00")],
        horizon_minutes=15,
        persist_artifacts=False,
    )

    assert captured["candidate_scope"] == "selected_only"
    assert captured["selection"] == replay_selection
    assert str(detail.iloc[0]["candidate_label"]) == module._qualify_control_candidate_label(
        replay_selection,
        "candidate_a",
    )


def test_replay_phase_detail_for_scope_reconstructs_full_family_when_stack_policy_has_no_selection(
    monkeypatch,
    tmp_path: Path,
):
    """Selection-less stack policies should rematerialize the parent phase family on rolling scopes."""
    captured: dict[str, object] = {}

    def fake_run_cached_rollout_evaluation(**kwargs):
        captured["selection"] = dict(kwargs["selection"])
        captured["candidate_scope"] = str(kwargs["candidate_scope"])
        return {
            "detail_by_origin": pd.DataFrame(
                [{"candidate_label": "candidate_a", "predicted_load": 1.0}]
            )
        }

    monkeypatch.setattr(
        module,
        "_representable_selection_origins",
        lambda **kwargs: [pd.Timestamp(value) for value in kwargs["origin_timestamps"]],
    )
    monkeypatch.setattr(module, "_run_cached_rollout_evaluation", fake_run_cached_rollout_evaluation)
    phase_payload = {
        "selection": {
            "resolution": "5min",
            "feature_set": "full",
            "model_label": "parent_model",
            "requested_candidate_label": "candidate_a",
        },
        "origin_policy": "phase_balanced",
        "selection_target": "next_lock_mae",
    }

    detail = module._replay_phase_detail_for_scope(
        phase_payload=phase_payload,
        phase_replay_metadata={"mode": "phase_bucket_portfolio", "selection": None},
        cache_root=None,
        temp_output_root=tmp_path / "portfolio",
        origin_timestamps=[pd.Timestamp("2026-03-01T00:00:00")],
        horizon_minutes=15,
        persist_artifacts=False,
    )

    assert captured["candidate_scope"] == "full_family"
    assert captured["selection"] == phase_payload["selection"]
    assert str(detail.iloc[0]["candidate_label"]) == module._qualify_control_candidate_label(
        phase_payload["selection"],
        "candidate_a",
    )


def test_replay_phase_detail_for_scope_filters_nonrepresentable_origins_before_replay(
    monkeypatch,
    tmp_path: Path,
):
    """Rolling phase replay should skip explicit origins the candidate family cannot represent."""
    captured: dict[str, object] = {}

    def fake_representable_selection_origins(**kwargs):
        origin_timestamps = [pd.Timestamp(value) for value in kwargs["origin_timestamps"]]
        return origin_timestamps[:1]

    def fake_run_cached_rollout_evaluation(**kwargs):
        captured["origin_timestamps"] = list(kwargs["origin_timestamps"])
        return {
            "detail_by_origin": pd.DataFrame(
                [{"candidate_label": "candidate_a", "predicted_load": 1.0}]
            )
        }

    monkeypatch.setattr(module, "_representable_selection_origins", fake_representable_selection_origins)
    monkeypatch.setattr(module, "_run_cached_rollout_evaluation", fake_run_cached_rollout_evaluation)
    phase_payload = {
        "selection": {
            "resolution": "5min",
            "feature_set": "full",
            "model_label": "parent_model",
            "requested_candidate_label": "candidate_a",
        },
        "origin_policy": "phase_balanced",
        "selection_target": "next_lock_mae",
    }
    origins = [
        pd.Timestamp("2026-03-01T00:00:00"),
        pd.Timestamp("2026-03-01T00:05:00"),
    ]

    module._replay_phase_detail_for_scope(
        phase_payload=phase_payload,
        phase_replay_metadata={"mode": "phase_bucket_portfolio", "selection": None},
        cache_root=None,
        temp_output_root=tmp_path / "portfolio_filtered",
        origin_timestamps=origins,
        horizon_minutes=15,
        persist_artifacts=False,
    )

    assert captured["origin_timestamps"] == origins[:1]


def test_apply_bucketed_series_updates_can_mix_hourly_and_candidate_segments():
    """Stack-aware bucket routing should be able to choose any precomputed candidate series per bucket."""
    origin = pd.Timestamp("2026-03-01T00:00:00")
    minute_index = pd.date_range(origin + pd.Timedelta(minutes=1), periods=30, freq="1min")
    hourly = pd.Series([10.0] * len(minute_index), index=minute_index, dtype=float)
    candidate_a = pd.Series([20.0] * len(minute_index), index=minute_index, dtype=float)
    candidate_b = pd.Series([30.0] * len(minute_index), index=minute_index, dtype=float)

    series = module._apply_bucketed_series_updates(
        hourly,
        series_by_candidate={
            "hourly": hourly,
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
        },
        candidate_by_bucket={0: "hourly", 15: "candidate_b"},
        update_origins=[origin, origin + pd.Timedelta(minutes=15)],
        horizon_minutes=15,
        bucket_minutes=15,
    )

    assert float(series.iloc[0]) == 10.0
    assert float(series.iloc[14]) == 10.0
    assert float(series.iloc[15]) == 30.0
    assert float(series.iloc[-1]) == 30.0


def test_apply_bucketed_series_updates_handles_duplicate_timestamp_labels():
    """Stack-aware bucket routing should work on overlapping-cycle minute grids with duplicate timestamps."""
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-03-01T00:01:00"),
            pd.Timestamp("2026-03-01T00:02:00"),
            pd.Timestamp("2026-03-01T00:01:00"),
            pd.Timestamp("2026-03-01T00:02:00"),
        ]
    )
    hourly = pd.Series([10.0, 10.0, 10.0, 10.0], index=idx, dtype=float)
    candidate = pd.Series([20.0, 20.0, 30.0, 30.0], index=idx, dtype=float)

    series = module._apply_bucketed_series_updates(
        hourly,
        series_by_candidate={"candidate": candidate},
        candidate_by_bucket={0: "candidate"},
        update_origins=[pd.Timestamp("2026-03-01T00:00:00")],
        horizon_minutes=2,
        bucket_minutes=15,
    )

    assert series.tolist() == [20.0, 20.0, 30.0, 30.0]


def test_phase_stack_candidate_benchmark_scope_emits_bucket_portfolio_candidate(monkeypatch):
    """Phase stack benchmarking should build a bucket portfolio from per-origin candidate metrics."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_selection_metric", "next_lock_mae")
    cycle_origin = pd.Timestamp("2026-03-01T00:00:00")
    minute_index = pd.date_range(cycle_origin + pd.Timedelta(minutes=1), periods=30, freq="1min")
    actual_load = [20.0] * 15 + [30.0] * 15
    minute_timeline = pd.DataFrame(
        {
            "cycle_origin_timestamp": [cycle_origin.isoformat()] * len(minute_index),
            "timestamp": minute_index,
            "actual_load": actual_load,
            "day_ahead_pred": [10.0] * len(minute_index),
            "hourly_pred": [10.0] * len(minute_index),
        }
    )
    candidate_detail = pd.DataFrame(
        [
            {
                "origin_timestamp": cycle_origin.isoformat(),
                "candidate_label": "candidate_a",
                "forecast_timestamp": timestamp.isoformat(),
                "predicted_load": 20.0,
            }
            for timestamp in minute_index[:15]
        ]
        + [
            {
                "origin_timestamp": (cycle_origin + pd.Timedelta(minutes=15)).isoformat(),
                "candidate_label": "candidate_b",
                "forecast_timestamp": timestamp.isoformat(),
                "predicted_load": 30.0,
            }
            for timestamp in minute_index[15:30]
        ]
    )
    candidate_metrics = pd.DataFrame(
        [
            {
                "origin_timestamp": cycle_origin.isoformat(),
                "candidate_label": "candidate_a",
                "next_lock_mae": 0.0,
            },
            {
                "origin_timestamp": cycle_origin.isoformat(),
                "candidate_label": "candidate_b",
                "next_lock_mae": 10.0,
            },
            {
                "origin_timestamp": (cycle_origin + pd.Timedelta(minutes=15)).isoformat(),
                "candidate_label": "candidate_a",
                "next_lock_mae": 10.0,
            },
            {
                "origin_timestamp": (cycle_origin + pd.Timedelta(minutes=15)).isoformat(),
                "candidate_label": "candidate_b",
                "next_lock_mae": 0.0,
            },
        ]
    )
    candidate_meta = pd.DataFrame(
        [
            {
                "candidate_label": "candidate_a",
                "candidate_type": "learned",
                "source_model_label": "model_a",
                "target_mode": "raw",
                "replay_pool_rank": 1.0,
                "replay_pool_source_type": "test",
                "replay_pool_source_run_id": "run_a",
                "replay_resolution": "1min",
                "replay_feature_set": "minimal",
                "replay_model_label": "model_a",
                "replay_run_dir": "outputs/test/run_a",
            },
            {
                "candidate_label": "candidate_b",
                "candidate_type": "learned",
                "source_model_label": "model_b",
                "target_mode": "raw",
                "replay_pool_rank": 2.0,
                "replay_pool_source_type": "test",
                "replay_pool_source_run_id": "run_b",
                "replay_resolution": "1min",
                "replay_feature_set": "minimal",
                "replay_model_label": "model_b",
                "replay_run_dir": "outputs/test/run_b",
            },
        ]
    )

    benchmark, _, _ = module._phase_stack_candidate_benchmark_scope(
        minute_timeline=minute_timeline,
        candidate_detail_by_origin=candidate_detail,
        candidate_metrics_by_origin=candidate_metrics,
        phase_origins=[cycle_origin, cycle_origin + pd.Timedelta(minutes=15)],
        phase_horizon=15,
        lock_interval=15,
        hourly_candidate_label="hourly",
        hourly_candidate_type="learned",
        hourly_source_model_label="hourly_model",
        candidate_meta=candidate_meta,
    )

    portfolio_row = benchmark.loc[
        benchmark["candidate_label"]
        .astype("string")
        .eq("phase_bucket_portfolio::origin_minute_policy")
    ]
    assert not portfolio_row.empty
    assert float(portfolio_row.iloc[0]["lock_mae"]) < float(
        benchmark.loc[benchmark["candidate_label"].astype("string").eq("hourly"), "lock_mae"].iloc[0]
    )
    assert benchmark["candidate_label"].astype("string").eq(
        "phase_bucket_portfolio::stack_origin_metric_policy"
    ).any()


def test_select_phase_stack_blend_parent_labels_caps_to_top_native_learned_candidates(monkeypatch):
    """Only the strongest native learned phase parents should expand into hourly-phase blends."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_selection_metric", "next_lock_mae")

    benchmark = pd.DataFrame(
        [
            {
                "candidate_label": "candidate_c",
                "candidate_type": "learned",
                "stack_candidate_family": "native_phase_candidate",
                "next_lock_mae": 12.0,
                "lock_mae": 30.0,
                "profile_shape_mae": 40.0,
                "minute_path_mae": 50.0,
            },
            {
                "candidate_label": "candidate_a",
                "candidate_type": "learned",
                "stack_candidate_family": "native_phase_candidate",
                "next_lock_mae": 5.0,
                "lock_mae": 20.0,
                "profile_shape_mae": 30.0,
                "minute_path_mae": 40.0,
            },
            {
                "candidate_label": "candidate_b",
                "candidate_type": "learned",
                "stack_candidate_family": "native_phase_candidate",
                "next_lock_mae": 8.0,
                "lock_mae": 25.0,
                "profile_shape_mae": 35.0,
                "minute_path_mae": 45.0,
            },
            {
                "candidate_label": "baseline_x",
                "candidate_type": "baseline",
                "stack_candidate_family": "native_phase_candidate",
                "next_lock_mae": 1.0,
                "lock_mae": 10.0,
                "profile_shape_mae": 20.0,
                "minute_path_mae": 30.0,
            },
            {
                "candidate_label": "candidate_a|stack_blend_w0.25",
                "candidate_type": "learned",
                "stack_candidate_family": "hourly_phase_blend",
                "next_lock_mae": 4.0,
                "lock_mae": 18.0,
                "profile_shape_mae": 28.0,
                "minute_path_mae": 38.0,
            },
        ]
    )

    labels = module._select_phase_stack_blend_parent_labels(benchmark, limit=2)

    assert labels == ["candidate_a", "candidate_b"]


def test_phase_stack_native_candidate_shortlist_limits_candidates_per_pool(monkeypatch):
    """Stage-10 should keep only the strongest native learned and baseline candidates per replay pool."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "control_promotion_scope", "held_out_evaluation")
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_native_learned_top_candidates_per_pool", 1)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_native_baseline_top_candidates_per_pool", 1)

    candidate_meta = pd.DataFrame(
        [
            {
                "candidate_label": "learned_pool1_best",
                "candidate_type": "learned",
                "replay_pool_rank": 1.0,
                "replay_pool_source_type": "rollout_registry",
                "replay_pool_source_run_id": "run_1",
                "replay_resolution": "5min",
                "replay_feature_set": "full",
                "replay_model_label": "model_pool1",
                "evaluation_selection_metric_value": 2.0,
                "evaluation_next_lock_mae": 2.0,
            },
            {
                "candidate_label": "learned_pool1_other",
                "candidate_type": "learned",
                "replay_pool_rank": 1.0,
                "replay_pool_source_type": "rollout_registry",
                "replay_pool_source_run_id": "run_1",
                "replay_resolution": "5min",
                "replay_feature_set": "full",
                "replay_model_label": "model_pool1",
                "evaluation_selection_metric_value": 5.0,
                "evaluation_next_lock_mae": 5.0,
            },
            {
                "candidate_label": "baseline_pool1_best",
                "candidate_type": "baseline",
                "replay_pool_rank": 1.0,
                "replay_pool_source_type": "rollout_registry",
                "replay_pool_source_run_id": "run_1",
                "replay_resolution": "5min",
                "replay_feature_set": "full",
                "replay_model_label": "model_pool1",
                "evaluation_selection_metric_value": 3.0,
                "evaluation_next_lock_mae": 3.0,
            },
            {
                "candidate_label": "baseline_pool1_other",
                "candidate_type": "baseline",
                "replay_pool_rank": 1.0,
                "replay_pool_source_type": "rollout_registry",
                "replay_pool_source_run_id": "run_1",
                "replay_resolution": "5min",
                "replay_feature_set": "full",
                "replay_model_label": "model_pool1",
                "evaluation_selection_metric_value": 9.0,
                "evaluation_next_lock_mae": 9.0,
            },
            {
                "candidate_label": "learned_pool2_best",
                "candidate_type": "learned",
                "replay_pool_rank": 2.0,
                "replay_pool_source_type": "rollout_registry",
                "replay_pool_source_run_id": "run_2",
                "replay_resolution": "5min",
                "replay_feature_set": "minimal",
                "replay_model_label": "model_pool2",
                "evaluation_selection_metric_value": 1.0,
                "evaluation_next_lock_mae": 1.0,
            },
            {
                "candidate_label": "learned_pool2_other",
                "candidate_type": "learned",
                "replay_pool_rank": 2.0,
                "replay_pool_source_type": "rollout_registry",
                "replay_pool_source_run_id": "run_2",
                "replay_resolution": "5min",
                "replay_feature_set": "minimal",
                "replay_model_label": "model_pool2",
                "evaluation_selection_metric_value": 4.0,
                "evaluation_next_lock_mae": 4.0,
            },
        ]
    )

    shortlisted = module._phase_stack_native_candidate_shortlist(candidate_meta)

    assert set(shortlisted["candidate_label"].astype("string")) == {
        "baseline_pool1_best",
        "learned_pool1_best",
        "learned_pool2_best",
    }


def test_phase_stack_candidate_benchmark_scope_limits_blend_expansion_to_top_parents(monkeypatch):
    """Stage-10 should avoid blending every native phase candidate when the parent cap is set."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_selection_metric", "next_lock_mae")
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_selection_metric", "next_lock_mae")
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_blend_weights", [0.25, 0.5])
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_blend_parent_top_candidates", 1)

    cycle_origin = pd.Timestamp("2026-03-01T00:00:00")
    minute_index = pd.date_range(cycle_origin + pd.Timedelta(minutes=1), periods=30, freq="1min")
    actual_load = [20.0] * 15 + [30.0] * 15
    minute_timeline = pd.DataFrame(
        {
            "cycle_origin_timestamp": [cycle_origin.isoformat()] * len(minute_index),
            "timestamp": minute_index,
            "actual_load": actual_load,
            "day_ahead_pred": [10.0] * len(minute_index),
            "hourly_pred": [10.0] * len(minute_index),
        }
    )
    candidate_detail = pd.DataFrame(
        [
            {
                "origin_timestamp": cycle_origin.isoformat(),
                "candidate_label": "candidate_a",
                "forecast_timestamp": timestamp.isoformat(),
                "predicted_load": 20.0,
            }
            for timestamp in minute_index[:15]
        ]
        + [
            {
                "origin_timestamp": (cycle_origin + pd.Timedelta(minutes=15)).isoformat(),
                "candidate_label": "candidate_b",
                "forecast_timestamp": timestamp.isoformat(),
                "predicted_load": 30.0,
            }
            for timestamp in minute_index[15:30]
        ]
    )
    candidate_metrics = pd.DataFrame(
        [
            {
                "origin_timestamp": cycle_origin.isoformat(),
                "candidate_label": "candidate_a",
                "next_lock_mae": 0.0,
            },
            {
                "origin_timestamp": cycle_origin.isoformat(),
                "candidate_label": "candidate_b",
                "next_lock_mae": 10.0,
            },
            {
                "origin_timestamp": (cycle_origin + pd.Timedelta(minutes=15)).isoformat(),
                "candidate_label": "candidate_a",
                "next_lock_mae": 10.0,
            },
            {
                "origin_timestamp": (cycle_origin + pd.Timedelta(minutes=15)).isoformat(),
                "candidate_label": "candidate_b",
                "next_lock_mae": 0.0,
            },
        ]
    )
    candidate_meta = pd.DataFrame(
        [
            {
                "candidate_label": "candidate_a",
                "candidate_type": "learned",
                "source_model_label": "model_a",
                "target_mode": "raw",
                "replay_pool_rank": 1.0,
                "replay_pool_source_type": "test",
                "replay_pool_source_run_id": "run_a",
                "replay_resolution": "1min",
                "replay_feature_set": "minimal",
                "replay_model_label": "model_a",
                "replay_run_dir": "outputs/test/run_a",
            },
            {
                "candidate_label": "candidate_b",
                "candidate_type": "learned",
                "source_model_label": "model_b",
                "target_mode": "raw",
                "replay_pool_rank": 2.0,
                "replay_pool_source_type": "test",
                "replay_pool_source_run_id": "run_b",
                "replay_resolution": "1min",
                "replay_feature_set": "minimal",
                "replay_model_label": "model_b",
                "replay_run_dir": "outputs/test/run_b",
            },
        ]
    )

    benchmark, _, _ = module._phase_stack_candidate_benchmark_scope(
        minute_timeline=minute_timeline,
        candidate_detail_by_origin=candidate_detail,
        candidate_metrics_by_origin=candidate_metrics,
        phase_origins=[cycle_origin, cycle_origin + pd.Timedelta(minutes=15)],
        phase_horizon=15,
        lock_interval=15,
        hourly_candidate_label="hourly",
        hourly_candidate_type="learned",
        hourly_source_model_label="hourly_model",
        candidate_meta=candidate_meta,
    )

    blend_rows = benchmark.loc[benchmark["stack_candidate_family"].astype("string").eq("hourly_phase_blend")].copy()
    assert len(blend_rows) == 2
    assert blend_rows["stack_blend_parent_candidate_label"].astype("string").eq("candidate_a").all()


def test_phase_stack_candidate_benchmark_scope_applies_native_shortlist_before_expansion(monkeypatch):
    """Stage-10 should trim native phase candidates before building stack blends and portfolios."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_selection_metric", "next_lock_mae")
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_selection_metric", "next_lock_mae")
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_native_learned_top_candidates_per_pool", 1)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_native_baseline_top_candidates_per_pool", 1)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_blend_weights", [0.25])
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_blend_parent_top_candidates", 1)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_bucket_policy_enabled", False)

    cycle_origin = pd.Timestamp("2026-03-01T00:00:00")
    minute_index = pd.date_range(cycle_origin + pd.Timedelta(minutes=1), periods=15, freq="1min")
    minute_timeline = pd.DataFrame(
        {
            "cycle_origin_timestamp": [cycle_origin.isoformat()] * len(minute_index),
            "timestamp": minute_index,
            "actual_load": [20.0] * len(minute_index),
            "day_ahead_pred": [10.0] * len(minute_index),
            "hourly_pred": [10.0] * len(minute_index),
        }
    )
    candidate_detail = pd.DataFrame(
        [
            {
                "origin_timestamp": cycle_origin.isoformat(),
                "candidate_label": "learned_best",
                "forecast_timestamp": timestamp.isoformat(),
                "predicted_load": 20.0,
            }
            for timestamp in minute_index
        ]
        + [
            {
                "origin_timestamp": cycle_origin.isoformat(),
                "candidate_label": "learned_other",
                "forecast_timestamp": timestamp.isoformat(),
                "predicted_load": 18.0,
            }
            for timestamp in minute_index
        ]
        + [
            {
                "origin_timestamp": cycle_origin.isoformat(),
                "candidate_label": "baseline_best",
                "forecast_timestamp": timestamp.isoformat(),
                "predicted_load": 19.0,
            }
            for timestamp in minute_index
        ]
        + [
            {
                "origin_timestamp": cycle_origin.isoformat(),
                "candidate_label": "baseline_other",
                "forecast_timestamp": timestamp.isoformat(),
                "predicted_load": 16.0,
            }
            for timestamp in minute_index
        ]
    )
    candidate_metrics = pd.DataFrame(
        [
            {"origin_timestamp": cycle_origin.isoformat(), "candidate_label": "learned_best", "next_lock_mae": 0.0},
            {"origin_timestamp": cycle_origin.isoformat(), "candidate_label": "learned_other", "next_lock_mae": 2.0},
            {"origin_timestamp": cycle_origin.isoformat(), "candidate_label": "baseline_best", "next_lock_mae": 1.0},
            {"origin_timestamp": cycle_origin.isoformat(), "candidate_label": "baseline_other", "next_lock_mae": 3.0},
        ]
    )
    candidate_meta = pd.DataFrame(
        [
            {
                "candidate_label": "learned_best",
                "candidate_type": "learned",
                "source_model_label": "model_best",
                "target_mode": "raw",
                "replay_pool_rank": 1.0,
                "replay_pool_source_type": "test",
                "replay_pool_source_run_id": "run_1",
                "replay_resolution": "5min",
                "replay_feature_set": "minimal",
                "replay_model_label": "model_best",
                "replay_run_dir": "outputs/test/run_1",
                "selection_metric_value": 0.0,
            },
            {
                "candidate_label": "learned_other",
                "candidate_type": "learned",
                "source_model_label": "model_other",
                "target_mode": "raw",
                "replay_pool_rank": 1.0,
                "replay_pool_source_type": "test",
                "replay_pool_source_run_id": "run_1",
                "replay_resolution": "5min",
                "replay_feature_set": "minimal",
                "replay_model_label": "model_other",
                "replay_run_dir": "outputs/test/run_1",
                "selection_metric_value": 2.0,
            },
            {
                "candidate_label": "baseline_best",
                "candidate_type": "baseline",
                "source_model_label": "baseline_best",
                "target_mode": "baseline",
                "replay_pool_rank": 1.0,
                "replay_pool_source_type": "test",
                "replay_pool_source_run_id": "run_1",
                "replay_resolution": "5min",
                "replay_feature_set": "minimal",
                "replay_model_label": "baseline_best",
                "replay_run_dir": "outputs/test/run_1",
                "selection_metric_value": 1.0,
            },
            {
                "candidate_label": "baseline_other",
                "candidate_type": "baseline",
                "source_model_label": "baseline_other",
                "target_mode": "baseline",
                "replay_pool_rank": 1.0,
                "replay_pool_source_type": "test",
                "replay_pool_source_run_id": "run_1",
                "replay_resolution": "5min",
                "replay_feature_set": "minimal",
                "replay_model_label": "baseline_other",
                "replay_run_dir": "outputs/test/run_1",
                "selection_metric_value": 3.0,
            },
        ]
    )

    benchmark, _, _ = module._phase_stack_candidate_benchmark_scope(
        minute_timeline=minute_timeline,
        candidate_detail_by_origin=candidate_detail,
        candidate_metrics_by_origin=candidate_metrics,
        phase_origins=[cycle_origin],
        phase_horizon=15,
        lock_interval=15,
        hourly_candidate_label="hourly",
        hourly_candidate_type="learned",
        hourly_source_model_label="hourly_model",
        candidate_meta=candidate_meta,
    )

    native_rows = benchmark.loc[
        benchmark["stack_candidate_family"].astype("string").eq("native_phase_candidate")
    ].copy()
    assert set(native_rows["candidate_label"].astype("string")) == {"learned_best", "baseline_best"}
    blend_rows = benchmark.loc[benchmark["stack_candidate_family"].astype("string").eq("hourly_phase_blend")].copy()
    assert set(blend_rows["stack_blend_parent_candidate_label"].astype("string")) == {"learned_best"}


def test_phase_stack_baseline_control_candidates_emit_calibrated_blend_rows(monkeypatch):
    """Stage-10 should add baseline-anchored phase control blends from held-out calibration."""
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_baseline_control_blend_enabled", True)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_baseline_control_top_candidates", 1)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_baseline_control_blend_weights", [0.2, 0.5, 0.8])
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_baseline_control_bucket_blend_enabled", True)
    monkeypatch.setitem(module.MULTIRES_FORECAST_CONTROL, "phase_stack_baseline_control_bucket_size_minutes", 5)

    cycle_origin = pd.Timestamp("2026-03-01T00:00:00")
    minute_index = pd.date_range(cycle_origin + pd.Timedelta(minutes=1), periods=15, freq="1min")
    minute_timeline = pd.DataFrame(
        {
            "cycle_origin_timestamp": [cycle_origin.isoformat()] * len(minute_index),
            "timestamp": minute_index,
            "actual_load": [15.0] * len(minute_index),
            "day_ahead_pred": [10.0] * len(minute_index),
            "hourly_pred": [10.0] * len(minute_index),
        }
    )
    hourly_row = {
        "candidate_label": "hourly",
        "candidate_type": "passthrough",
        "source_model_label": "hourly",
        "target_mode": "hourly_passthrough",
        "minute_path_mae": 5.0,
        "minute_path_mae_pct": 10.0,
        "lock_mae": 5.0,
        "lock_mae_pct": 10.0,
        "profile_shape_mae": 5.0,
        "profile_shape_mae_pct": 10.0,
        "energy_mae": 5.0,
        "energy_mae_pct": 10.0,
        "cycle_n": 1,
        "lock_mae_p50": 5.0,
        "lock_mae_p90": 5.0,
        "profile_shape_mae_p50": 5.0,
        "profile_shape_mae_p90": 5.0,
        "minute_path_mae_p50": 5.0,
        "minute_path_mae_p90": 5.0,
        "lock_gain_vs_hourly": 0.0,
        "lock_gain_pct_vs_hourly": 0.0,
        "profile_degrade_vs_hourly": 0.0,
        "profile_degrade_pct_vs_hourly": 0.0,
        "meets_lock_gain_rule": False,
        "meets_profile_rule": False,
        "meets_stack_guard": False,
        "stack_candidate_policy": "hourly_passthrough",
        "selection_metric_name": "lock_mae",
        "selection_metric_value": 5.0,
        "selection_metric_pct": 10.0,
        "replay_pool_rank": 0.0,
        "replay_pool_source_type": "hourly_passthrough",
        "replay_pool_source_run_id": "",
        "replay_resolution": "",
        "replay_feature_set": "",
        "replay_model_label": "hourly",
        "replay_run_dir": "",
        "stack_blend_weight": float("nan"),
        "stack_blend_parent_candidate_label": "",
        "stack_reference_candidate_label": "",
        "stack_candidate_family": "hourly_passthrough",
        "stack_bucket_policy_json": "",
        "stack_bucket_weight_json": "",
        "stack_bucket_granularity_minutes": float("nan"),
    }
    baseline_row = {
        **hourly_row,
        "candidate_label": "1min/minimal/persistence",
        "candidate_type": "baseline",
        "source_model_label": "persistence",
        "target_mode": "baseline",
        "replay_pool_rank": 1.0,
        "replay_pool_source_type": "test",
        "replay_pool_source_run_id": "run_1",
        "replay_resolution": "1min",
        "replay_feature_set": "minimal",
        "replay_model_label": "persistence",
        "stack_candidate_family": "native_phase_candidate",
    }
    learned_row = {
        **baseline_row,
        "candidate_label": "1min/minimal/model::raw",
        "candidate_type": "learned",
        "source_model_label": "model",
        "target_mode": "raw",
        "replay_pool_rank": 2.0,
        "replay_model_label": "model",
    }
    calibration_benchmark = pd.DataFrame([hourly_row, baseline_row, learned_row])
    evaluation_benchmark = calibration_benchmark.copy()
    hourly_series = pd.Series([10.0] * len(minute_index), index=minute_index, dtype=float)
    baseline_series = pd.Series([10.0] * len(minute_index), index=minute_index, dtype=float)
    learned_series = pd.Series([20.0] * len(minute_index), index=minute_index, dtype=float)

    (
        calibration_benchmark,
        evaluation_benchmark,
        calibration_predictions,
        evaluation_predictions,
        _,
        _,
    ) = module._phase_stack_baseline_control_candidates(
        calibration_minute_timeline=minute_timeline,
        evaluation_minute_timeline=minute_timeline,
        lock_interval=15,
        calibration_benchmark=calibration_benchmark,
        evaluation_benchmark=evaluation_benchmark,
        calibration_predictions={
            "hourly": hourly_series,
            "1min/minimal/persistence": baseline_series,
            "1min/minimal/model::raw": learned_series,
        },
        evaluation_predictions={
            "hourly": hourly_series,
            "1min/minimal/persistence": baseline_series,
            "1min/minimal/model::raw": learned_series,
        },
        calibration_summaries={},
        evaluation_summaries={},
    )

    assert calibration_benchmark["candidate_label"].astype("string").str.contains(
        "baseline_control_blend"
    ).any()
    assert evaluation_benchmark["candidate_label"].astype("string").str.contains(
        "baseline_control_bucket_blend"
    ).any()
    selected_label = "1min/minimal/model::raw|baseline_control_blend_w0.50"
    assert selected_label in calibration_predictions
    assert float(calibration_predictions[selected_label].iloc[0]) == 15.0


def test_replay_control_pool_candidate_reuses_one_union_replay_for_phase_scope(monkeypatch, tmp_path):
    """Phase challenger pools should replay once across the union of requested origins."""
    calls: list[dict[str, object]] = []

    def _fake_run_cached_rollout_evaluation(**kwargs):
        calls.append(dict(kwargs))
        return {
            "by_origin": pd.DataFrame(
                {
                    "origin_timestamp": [
                        pd.Timestamp("2026-03-01T00:00:00").isoformat(),
                        pd.Timestamp("2026-03-01T00:15:00").isoformat(),
                    ],
                    "candidate_label": ["model::raw", "model::raw"],
                }
            ),
            "detail_by_origin": pd.DataFrame(
                {
                    "origin_timestamp": [
                        pd.Timestamp("2026-03-01T00:00:00").isoformat(),
                        pd.Timestamp("2026-03-01T00:15:00").isoformat(),
                    ],
                    "candidate_label": ["model::raw", "model::raw"],
                    "forecast_timestamp": [
                        pd.Timestamp("2026-03-01T00:01:00").isoformat(),
                        pd.Timestamp("2026-03-01T00:16:00").isoformat(),
                    ],
                    "predicted_load": [1.0, 2.0],
                }
            ),
            "run_dir": tmp_path / "demo",
            "replay_cache_status": "hit",
            "selected_origins": pd.DataFrame(
                {
                    "origin_timestamp": [
                        pd.Timestamp("2026-03-01T00:00:00").isoformat(),
                        pd.Timestamp("2026-03-01T00:15:00").isoformat(),
                    ]
                }
            ),
        }

    monkeypatch.setattr(module, "_run_cached_rollout_evaluation", _fake_run_cached_rollout_evaluation)

    result = module._replay_control_pool_candidate(
        cache_root=None,
        temp_root=tmp_path,
        layer_role="phase",
        pool_rank=1,
        pool_item={"selection": {"resolution": "1min"}, "pool_source_type": "demo", "pool_source_run_id": ""},
        candidate_scope="selected_only",
        horizon_minutes=15,
        origin_policy="phase_balanced",
        selection_target="next_lock_mae",
        benchmark_origins=[pd.Timestamp("2026-03-01T00:00:00")],
        evaluation_benchmark_origins=[pd.Timestamp("2026-03-01T00:15:00")],
    )

    assert len(calls) == 1
    assert bool(calls[0]["capture_path_details"]) is True
    assert sorted(pd.to_datetime(calls[0]["origin_timestamps"]).tolist()) == [
        pd.Timestamp("2026-03-01T00:00:00"),
        pd.Timestamp("2026-03-01T00:15:00"),
    ]
    assert result["benchmark_result"]["by_origin"]["origin_timestamp"].tolist() == [
        pd.Timestamp("2026-03-01T00:00:00").isoformat()
    ]
    assert result["evaluation_result"]["by_origin"]["origin_timestamp"].tolist() == [
        pd.Timestamp("2026-03-01T00:15:00").isoformat()
    ]


def test_build_runtime_profile_summary_surfaces_longest_step():
    """Stage-10 runtime summaries should expose the longest bottleneck step clearly."""
    runtime_profile = pd.DataFrame(
        [
            {"step": "prepare_control_scope", "category": "setup", "duration_seconds": 2.0},
            {"step": "replay_phase_layer", "category": "replay", "duration_seconds": 12.5},
            {"step": "evaluate_rolling_control_scope", "category": "evaluation", "duration_seconds": 5.5},
            {"step": "write_stage_outputs", "category": "artifacts", "duration_seconds": 1.0},
        ]
    )

    summary = module._build_runtime_profile_summary(runtime_profile, wall_clock_seconds=22.0)

    assert summary["step_count"] == 4
    assert summary["longest_step"] == "replay_phase_layer"
    assert float(summary["longest_step_seconds"]) == 12.5
    assert float(summary["replay_seconds"]) == 12.5
    assert float(summary["evaluation_seconds"]) == 5.5
    assert float(summary["artifacts_seconds"]) == 1.0
    assert float(summary["setup_seconds"]) == 2.0


def test_build_current_evidence_index_includes_exact_rolling_and_repo_level_rows(monkeypatch, tmp_path):
    """The Stage-10 evidence index should summarize current exact, rolling, and upstream winners."""
    performance_root = tmp_path / "005_performance"
    horizon_root = tmp_path / "009_horizon_curve"
    (performance_root / "latest").mkdir(parents=True, exist_ok=True)
    (horizon_root / "latest").mkdir(parents=True, exist_ok=True)
    (performance_root / "latest" / "holdout_evaluation.csv").write_text(
        "\n".join(
            [
                "candidate_label,candidate_type,mae,mae_pct",
                "persistence,baseline,173.724099,8.380502",
                "curated_ramp/hgb-frontier-lr010-leaf100/residual+blend,promoted_learned,174.891813,8.436833",
            ]
        ),
        encoding="utf-8",
    )
    (horizon_root / "latest" / "horizon_curve_summary.csv").write_text(
        "\n".join(
            [
                "horizon_minutes,selection_target,candidate_label,learned_next_lock_mae,learned_next_lock_mae_pct,learned_profile_shape_mae,learned_profile_shape_mae_pct",
                "15,next_lock_mae,hgb-balanced::phase_bucket_next_lock_policy,266.837858,9.333856,247.848341,8.669613",
                "1440,profile_shape_mae,hgb-balanced::raw,409.047215,19.456219,717.777613,36.245099",
            ]
        ),
        encoding="utf-8",
    )

    def _fake_preferred_output_path(path: Path) -> Path:
        if path.name == "005_performance":
            return performance_root
        if path.name == "009_horizon_curve":
            return horizon_root
        return tmp_path / path.name

    monkeypatch.setattr(module, "preferred_output_path", _fake_preferred_output_path)

    summary = pd.DataFrame(
        [
            {
                "layer": "day_ahead_frozen",
                "role": "day_ahead",
                "cycle_n": 4,
                "minute_path_mae": 10.0,
                "minute_path_mae_pct": 1.0,
                "lock_mae": 9.0,
                "lock_mae_pct": 0.9,
                "profile_shape_mae": 8.0,
                "profile_shape_mae_pct": 0.8,
                "energy_mae": 7.0,
                "energy_mae_pct": 0.7,
            },
            {
                "layer": "after_hourly_updates",
                "role": "hourly",
                "cycle_n": 4,
                "minute_path_mae": 8.0,
                "minute_path_mae_pct": 0.8,
                "lock_mae": 6.0,
                "lock_mae_pct": 0.6,
                "profile_shape_mae": 7.0,
                "profile_shape_mae_pct": 0.7,
                "energy_mae": 6.0,
                "energy_mae_pct": 0.6,
            },
            {
                "layer": "after_phase_updates",
                "role": "phase",
                "cycle_n": 4,
                "minute_path_mae": 7.0,
                "minute_path_mae_pct": 0.7,
                "lock_mae": 5.0,
                "lock_mae_pct": 0.5,
                "profile_shape_mae": 6.0,
                "profile_shape_mae_pct": 0.6,
                "energy_mae": 5.0,
                "energy_mae_pct": 0.5,
            },
            {
                "layer": "after_nowcast_updates",
                "role": "nowcast",
                "cycle_n": 4,
                "minute_path_mae": 4.0,
                "minute_path_mae_pct": 0.4,
                "lock_mae": 3.0,
                "lock_mae_pct": 0.3,
                "profile_shape_mae": 4.0,
                "profile_shape_mae_pct": 0.4,
                "energy_mae": 3.0,
                "energy_mae_pct": 0.3,
            },
        ]
    )
    rolling_scope_summary = pd.DataFrame(
        [
            {
                "scope": "rolling_evaluation",
                "layer": "after_hourly_updates",
                "role": "hourly",
                "cycle_n": 12,
                "lock_mae": 6.5,
                "lock_mae_p50": 6.0,
                "lock_mae_p90": 7.0,
                "lock_mae_pct": 0.65,
                "profile_shape_mae": 7.5,
                "profile_shape_mae_p50": 7.0,
                "profile_shape_mae_p90": 8.0,
                "profile_shape_mae_pct": 0.75,
            }
        ]
    )
    rolling_layer_inference = pd.DataFrame(
        [
            {
                "scope": "rolling_evaluation",
                "comparison_label": "hourly_vs_day_ahead",
                "metric_name": "lock_mae",
                "gain_metric": 2.5,
                "gain_metric_ci_low": 1.0,
                "gain_metric_ci_high": 3.0,
                "two_sided_p": 0.04,
            }
        ]
    )
    refresh_summary = pd.DataFrame(
        [
            {"scenario": "frozen_day_ahead", "lock_mae": 9.0, "lock_mae_pct": 0.9, "profile_shape_mae": 8.0, "profile_shape_mae_pct": 0.8},
            {"scenario": "unconditional_refresh", "lock_mae": 7.0, "lock_mae_pct": 0.7, "profile_shape_mae": 6.0, "profile_shape_mae_pct": 0.6},
        ]
    )
    policy = {
        "day_ahead": {"candidate_label": "10min/minimal/hgb-balanced::raw"},
        "hourly": {"candidate_label": "10min/minimal/hgb-balanced::hybrid_phase_gate"},
        "phase": {"candidate_label": "5min/full/persistence"},
        "nowcast_anchor": {"candidate_label": "persistence"},
        "day_ahead_refresh": {"recommended_policy": "unconditional_refresh", "candidate_label": "hybrid_workday_residual"},
    }

    index = module._build_current_evidence_index(
        summary=summary,
        policy=policy,
        rolling_scope_summary=rolling_scope_summary,
        rolling_layer_inference=rolling_layer_inference,
        refresh_summary=refresh_summary,
        rolling_refresh_summary=pd.DataFrame(),
    )

    assert {"stage10_exact_control", "stage10_rolling_benchmark", "stage10_rolling_inference", "stage5_holdout", "stage8_horizon_curve"}.issubset(
        set(index["section"])
    )
    assert "10min/minimal/hgb-balanced::raw" in set(index["candidate_label"])
    assert "persistence" in set(index["candidate_label"])
