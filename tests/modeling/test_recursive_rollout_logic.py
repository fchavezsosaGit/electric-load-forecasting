"""Unit tests for recursive rollout helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modeling.common import build_model_catalog, train_model
from modeling.multires import (
    actual_path,
    anchored_workday_path,
    avg_workday_path,
    build_workday_profile,
    blend_candidate_paths,
    build_causal_feature_frame,
    compare_recursive_paths,
    persistence_path,
    recursive_predict_path,
    recursive_predict_residual_path,
)


def _load_recursive_rollout_module():
    """Load the recursive rollout script as an importable module for helper-level tests."""
    path = Path("scripts/modeling/recursive_rollout.py").resolve()
    spec = importlib.util.spec_from_file_location("test_recursive_rollout_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load recursive rollout module for testing.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_base(days: int = 3, freq: str = "5min") -> pd.DataFrame:
    """Create a deterministic load series that is rich enough for rollout helper tests."""
    timestamps = pd.date_range("2025-12-01 00:00:00", periods=days * 24 * 12, freq=freq)
    signal = 100.0 + np.sin(np.arange(len(timestamps)) / 12.0) * 10.0
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "avg_load": signal.astype(float),
            "day_class": ["full"] * len(timestamps),
            "day_idx": np.repeat(np.arange(1, days + 1), 24 * 12),
        }
    )


def _write_rollout_history_run(
    output_root: Path,
    *,
    run_id: str,
    resolution: str,
    feature_set: str,
    model_label: str,
    horizon_minutes: int,
    origin_policy: str,
    selection_target: str,
    learned_endpoint_mae: float,
    learned_path_mae: float,
    persistence_endpoint_mae: float,
    persistence_path_mae: float,
    avg_workday_endpoint_mae: float,
    avg_workday_path_mae: float,
    learned_phase_mean_mae: float | None = None,
    persistence_phase_mean_mae: float | None = None,
    avg_workday_phase_mean_mae: float | None = None,
    generated_at_utc: str = "2026-03-09T19:47:12.097122+00:00",
    origins_per_run: int = 8,
) -> None:
    """Write a minimal historical rollout run so selection-registry logic can be exercised."""
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "stage": "007_rollout",
                "mode": "candidate",
                "strategy": "recursive",
                "horizon_minutes": horizon_minutes,
                "origins_per_run": origins_per_run,
                "origin_policy": origin_policy,
                "generated_at_utc": generated_at_utc,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "selection_context.json").write_text(
        json.dumps(
            {
                "resolution": resolution,
                "feature_set": feature_set,
                "model_label": model_label,
                "forecast_strategy": "recursive",
                "requested_horizon_minutes": horizon_minutes,
                "requested_origin_policy": origin_policy,
                "selection_target": selection_target,
                "selection_source": "multires.toml",
                "selection_policy": "multires.toml",
                "selection_run_id": None,
                "selection_run_stage": None,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "candidate_label": model_label,
                "endpoint_mae": learned_endpoint_mae,
                "endpoint_rmse": learned_endpoint_mae,
                "path_mae": learned_path_mae,
                "phase_mean_mae": learned_path_mae if learned_phase_mean_mae is None else learned_phase_mean_mae,
                "mean_coverage": 1.0,
                "origin_n": origins_per_run,
            },
            {
                "candidate_label": "persistence",
                "endpoint_mae": persistence_endpoint_mae,
                "endpoint_rmse": persistence_endpoint_mae,
                "path_mae": persistence_path_mae,
                "phase_mean_mae": (
                    persistence_path_mae
                    if persistence_phase_mean_mae is None
                    else persistence_phase_mean_mae
                ),
                "mean_coverage": 1.0,
                "origin_n": origins_per_run,
            },
            {
                "candidate_label": "avg_workday",
                "endpoint_mae": avg_workday_endpoint_mae,
                "endpoint_rmse": avg_workday_endpoint_mae,
                "path_mae": avg_workday_path_mae,
                "phase_mean_mae": (
                    avg_workday_path_mae
                    if avg_workday_phase_mean_mae is None
                    else avg_workday_phase_mean_mae
                ),
                "mean_coverage": 1.0,
                "origin_n": origins_per_run,
            },
        ]
    ).to_csv(run_dir / "recursive_rollout_metrics.csv", index=False)
    endpoint_baseline_winner = (
        "persistence" if persistence_endpoint_mae <= avg_workday_endpoint_mae else "avg_workday"
    )
    path_baseline_winner = "persistence" if persistence_path_mae <= avg_workday_path_mae else "avg_workday"
    learned_phase = learned_path_mae if learned_phase_mean_mae is None else learned_phase_mean_mae
    persistence_phase = (
        persistence_path_mae if persistence_phase_mean_mae is None else persistence_phase_mean_mae
    )
    avg_workday_phase = (
        avg_workday_path_mae if avg_workday_phase_mean_mae is None else avg_workday_phase_mean_mae
    )
    endpoint_winner = (
        model_label if learned_endpoint_mae <= min(persistence_endpoint_mae, avg_workday_endpoint_mae) else endpoint_baseline_winner
    )
    path_winner = (
        model_label if learned_path_mae <= min(persistence_path_mae, avg_workday_path_mae) else path_baseline_winner
    )
    phase_baseline_winner = "persistence" if persistence_phase <= avg_workday_phase else "avg_workday"
    phase_winner = model_label if learned_phase <= min(persistence_phase, avg_workday_phase) else phase_baseline_winner
    pd.DataFrame(
        [
            {
                "selection_target": "endpoint_mae",
                "winner_candidate_label": endpoint_winner,
                "winner_metric_value": min(learned_endpoint_mae, persistence_endpoint_mae, avg_workday_endpoint_mae),
                "supporting_endpoint_mae": (
                    learned_endpoint_mae
                    if endpoint_winner == model_label
                    else persistence_endpoint_mae if endpoint_winner == "persistence" else avg_workday_endpoint_mae
                ),
                "supporting_path_mae": (
                    learned_path_mae
                    if endpoint_winner == model_label
                    else persistence_path_mae if endpoint_winner == "persistence" else avg_workday_path_mae
                ),
                "origin_n": origins_per_run,
                "decision_reason": "Lowest endpoint MAE across rollout candidates.",
            },
            {
                "selection_target": "path_mae",
                "winner_candidate_label": path_winner,
                "winner_metric_value": min(learned_path_mae, persistence_path_mae, avg_workday_path_mae),
                "supporting_endpoint_mae": (
                    learned_endpoint_mae
                    if path_winner == model_label
                    else persistence_endpoint_mae if path_winner == "persistence" else avg_workday_endpoint_mae
                ),
                "supporting_path_mae": (
                    learned_path_mae
                    if path_winner == model_label
                    else persistence_path_mae if path_winner == "persistence" else avg_workday_path_mae
                ),
                "origin_n": origins_per_run,
                "decision_reason": "Lowest path MAE across rollout candidates.",
            },
            {
                "selection_target": "phase_mean_mae",
                "winner_candidate_label": phase_winner,
                "winner_metric_value": min(learned_phase, persistence_phase, avg_workday_phase),
                "supporting_endpoint_mae": (
                    learned_endpoint_mae
                    if phase_winner == model_label
                    else persistence_endpoint_mae if phase_winner == "persistence" else avg_workday_endpoint_mae
                ),
                "supporting_path_mae": (
                    learned_path_mae
                    if phase_winner == model_label
                    else persistence_path_mae if phase_winner == "persistence" else avg_workday_path_mae
                ),
                "origin_n": origins_per_run,
                "decision_reason": "Lowest phase-average MAE across rollout candidates.",
            },
        ]
    ).to_csv(run_dir / "rollout_selection_summary.csv", index=False)


def test_selected_rollout_candidate_label_prefers_requested_override():
    """Direct replay overrides should win over sweep labels when both are present."""
    module = _load_recursive_rollout_module()

    label = module._selected_rollout_candidate_label(
        {
            "requested_candidate_label": "hgb-balanced::raw",
            "sweep_candidate_label": "ignored::candidate",
        }
    )

    assert label == "hgb-balanced::raw"


def test_requested_rollout_source_labels_expand_phase_bucket_policy_sources(monkeypatch):
    """Phase-bucket replays should request the mapped source labels instead of the selector label itself."""
    module = _load_recursive_rollout_module()
    monkeypatch.setattr(
        module,
        "_load_sweep_phase_policy_candidate",
        lambda _selection: {
            "candidate_label": "hgb-balanced::phase_bucket_next_lock_policy",
            "phase_bucket_mapping": {
                "0": "hgb-balanced::raw",
                "300": "hgb-balanced::anchored_workday_residual",
            },
        },
    )

    labels, payload = module._requested_rollout_source_labels(
        {"sweep_candidate_label": "hgb-balanced::phase_bucket_next_lock_policy"},
        requested_candidate_label="hgb-balanced::phase_bucket_next_lock_policy",
    )

    assert labels == {"hgb-balanced::raw", "hgb-balanced::anchored_workday_residual"}
    assert payload["candidate_label"] == "hgb-balanced::phase_bucket_next_lock_policy"


def test_recursive_predict_path_returns_expected_horizon_shape():
    """Recursive raw prediction should emit one ordered forecast row per requested horizon step."""
    base = _synthetic_base()
    frame = build_causal_feature_frame(base, "5min")
    train_df = frame.loc[frame["day_idx"].eq(1)].copy()
    trained = train_model(train_df, ["workday", "hour", "lag_1"], build_model_catalog()["ridge-medium"])
    origin_position = 24
    history = base.iloc[: origin_position + 1].set_index("timestamp")["avg_load"].astype(float)
    day_lookup = {timestamp.normalize(): "full" for timestamp in base["timestamp"]}

    path = recursive_predict_path(
        trained=trained,
        history=history,
        origin_timestamp=pd.Timestamp(base.iloc[origin_position]["timestamp"]),
        horizon_steps=3,
        resolution="5min",
        day_class_lookup=day_lookup,
    )

    assert len(path) == 3
    assert path["timestamp"].is_monotonic_increasing
    assert path["y_pred"].notna().all()


def test_recursive_predict_residual_path_returns_expected_horizon_shape():
    """Residual recursion against the workday baseline should preserve forecast shape and ordering."""
    base = _synthetic_base(days=4)
    frame = build_causal_feature_frame(base, "5min")
    train_df = frame.loc[frame["day_idx"].between(1, 2)].copy()
    feature_columns = [
        "workday",
        "hour",
        "lag_1",
        "avg_workday_baseline",
        "profile_residual_lag_1",
        "previous_day_residual",
    ]
    residual_train = train_df.copy()
    residual_train["avg_load"] = residual_train["avg_load"] - residual_train["avg_workday_baseline"]
    trained = train_model(residual_train, feature_columns, build_model_catalog()["ridge-medium"])
    profile = build_workday_profile(train_df)
    origin_position = 24 * 12 * 2
    history = base.iloc[: origin_position + 1].set_index("timestamp")["avg_load"].astype(float)
    day_lookup = {timestamp.normalize(): "full" for timestamp in base["timestamp"]}

    path = recursive_predict_residual_path(
        trained=trained,
        history=history,
        origin_timestamp=pd.Timestamp(base.iloc[origin_position]["timestamp"]),
        horizon_steps=3,
        resolution="5min",
        day_class_lookup=day_lookup,
        residual_baseline="avg_workday",
        profile=profile,
    )

    assert len(path) == 3
    assert path["timestamp"].is_monotonic_increasing
    assert path["y_pred"].notna().all()


def test_recursive_predict_residual_path_supports_persistence_baseline():
    """Residual recursion should also work when persistence is the correction baseline."""
    base = _synthetic_base(days=4)
    frame = build_causal_feature_frame(base, "5min")
    train_df = frame.loc[frame["day_idx"].between(1, 2)].copy()
    feature_columns = ["workday", "hour", "lag_1"]
    residual_train = train_df.copy()
    residual_train["avg_load"] = residual_train["avg_load"] - residual_train["lag_1"]
    trained = train_model(residual_train, feature_columns, build_model_catalog()["ridge-medium"])
    profile = build_workday_profile(train_df)
    origin_position = 24 * 12 * 2
    history = base.iloc[: origin_position + 1].set_index("timestamp")["avg_load"].astype(float)
    day_lookup = {timestamp.normalize(): "full" for timestamp in base["timestamp"]}

    path = recursive_predict_residual_path(
        trained=trained,
        history=history,
        origin_timestamp=pd.Timestamp(base.iloc[origin_position]["timestamp"]),
        horizon_steps=3,
        resolution="5min",
        day_class_lookup=day_lookup,
        residual_baseline="persistence",
        profile=profile,
    )

    assert len(path) == 3
    assert path["timestamp"].is_monotonic_increasing
    assert path["y_pred"].notna().all()


def test_recursive_predict_residual_path_supports_anchored_workday_baseline():
    """Residual recursion should support anchored-workday corrections with the required features."""
    base = _synthetic_base(days=4)
    frame = build_causal_feature_frame(base, "5min")
    train_df = frame.loc[frame["day_idx"].between(1, 2)].copy()
    feature_columns = [
        "workday",
        "hour",
        "lag_1",
        "anchored_workday_baseline",
        "profile_residual_lag_1",
        "previous_day_residual",
    ]
    residual_train = train_df.copy()
    residual_train["avg_load"] = residual_train["avg_load"] - residual_train["anchored_workday_baseline"]
    trained = train_model(residual_train, feature_columns, build_model_catalog()["ridge-medium"])
    profile = build_workday_profile(train_df)
    origin_position = 24 * 12 * 2
    history = base.iloc[: origin_position + 1].set_index("timestamp")["avg_load"].astype(float)
    day_lookup = {timestamp.normalize(): "full" for timestamp in base["timestamp"]}

    path = recursive_predict_residual_path(
        trained=trained,
        history=history,
        origin_timestamp=pd.Timestamp(base.iloc[origin_position]["timestamp"]),
        horizon_steps=3,
        resolution="5min",
        day_class_lookup=day_lookup,
        residual_baseline="anchored_workday",
        profile=profile,
    )

    assert len(path) == 3
    assert path["timestamp"].is_monotonic_increasing
    assert path["y_pred"].notna().all()


def test_recursive_predict_residual_path_supports_hybrid_workday_baseline():
    """Residual recursion should support hybrid-workday corrections over the full recursive path."""
    module = _load_recursive_rollout_module()
    base = _synthetic_base(days=4)
    frame = build_causal_feature_frame(base, "5min")
    train_df = frame.loc[frame["day_idx"].between(1, 2)].copy()
    feature_columns = [
        "workday",
        "hour",
        "lag_1",
        "anchored_workday_baseline",
        "profile_residual_lag_1",
        "previous_day_residual",
    ]
    persistence_weight = float(module.MULTIRES_HYBRID["persistence_weight_start"])
    residual_train = train_df.copy()
    hybrid_baseline = (
        persistence_weight * residual_train["lag_1"]
        + (1.0 - persistence_weight) * residual_train["anchored_workday_baseline"]
    )
    residual_train["avg_load"] = residual_train["avg_load"] - hybrid_baseline
    trained = train_model(residual_train, feature_columns, build_model_catalog()["ridge-medium"])
    profile = build_workday_profile(train_df)
    origin_position = 24 * 12 * 2
    history = base.iloc[: origin_position + 1].set_index("timestamp")["avg_load"].astype(float)
    day_lookup = {timestamp.normalize(): "full" for timestamp in base["timestamp"]}

    path = recursive_predict_residual_path(
        trained=trained,
        history=history,
        origin_timestamp=pd.Timestamp(base.iloc[origin_position]["timestamp"]),
        horizon_steps=3,
        resolution="5min",
        day_class_lookup=day_lookup,
        residual_baseline="hybrid_workday",
        profile=profile,
    )

    assert len(path) == 3
    assert path["timestamp"].is_monotonic_increasing
    assert path["y_pred"].notna().all()


def test_compare_recursive_paths_produces_endpoint_metrics():
    """Path comparison should return endpoint, path, and phase-aware error metrics."""
    base = _synthetic_base()
    origin_position = 24
    history = base.iloc[: origin_position + 1].set_index("timestamp")["avg_load"].astype(float)
    actual = actual_path(base, origin_position=origin_position, horizon_steps=4)
    persistence = persistence_path(
        history,
        origin_timestamp=pd.Timestamp(base.iloc[origin_position]["timestamp"]),
        horizon_steps=4,
        resolution="5min",
    )

    compared = compare_recursive_paths(actual, {"persistence": persistence})

    assert list(compared["candidate_label"]) == ["persistence"]
    assert compared.iloc[0]["endpoint_abs_error"] >= 0.0
    assert compared.iloc[0]["path_mae"] >= 0.0
    assert compared.iloc[0]["phase_mean_abs_error"] >= 0.0


def test_rollout_blend_end_weights_refine_around_historical_best(monkeypatch):
    """Blend-weight proposals should seed around the historically strongest saved blend endpoint."""
    module = _load_recursive_rollout_module()
    registry = pd.DataFrame(
        [
            {
                "run_id": "20260310T000000000000Z",
                "generated_at_utc": pd.Timestamp("2026-03-10T00:00:00Z"),
                "resolution": "1min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
                "learned_candidate_label": "hgb-balanced::persistence_raw_blend_e35",
                "horizon_minutes": 15,
                "origin_policy": "billing_aligned",
                "selection_target": "phase_mean_mae",
                "learned_phase_mean_mae": 360.0,
                "learned_path_mae": 460.0,
                "beats_best_baseline_phase": True,
                "beats_persistence_phase": True,
            }
        ]
    )
    monkeypatch.setattr(module, "_build_rollout_registry_snapshot", lambda _root: registry)

    weights = module._rollout_blend_end_weights(
        0.20,
        blend_family="persistence_raw_blend",
        output_root=Path("outputs/007_rollout/commercial_facility"),
        resolution="1min",
        feature_set="minimal",
        model_label="hgb-balanced",
        horizon_minutes=15,
        origin_policy="billing_aligned",
        selection_target="phase_mean_mae",
        alternate_end=0.35,
    )

    assert 0.20 in weights
    assert 0.35 in weights
    assert 0.30 in weights
    assert 0.40 in weights
    assert len(weights) <= module.MULTIRES_ROLLOUT_LEARNED_BLENDS["max_weights_per_family"]


def test_hybrid_phase_gate_weight_uses_phase_alignment():
    """Phase-gate weights should switch according to quarter-hour alignment buckets."""
    module = _load_recursive_rollout_module()

    aligned = module._hybrid_phase_gate_weight(pd.Timestamp("2025-12-26 00:00:00"))
    non_aligned = module._hybrid_phase_gate_weight(pd.Timestamp("2025-12-26 00:05:00"))
    ten_minute_offset = module._hybrid_phase_gate_weight(pd.Timestamp("2025-12-26 00:10:00"))
    unmatched_offset = module._hybrid_phase_gate_weight(pd.Timestamp("2025-12-26 00:02:00"))

    assert aligned == module.MULTIRES_ROLLOUT_LEARNED_BLENDS["hybrid_phase_gate_bucket_weights"][0]
    assert non_aligned == module.MULTIRES_ROLLOUT_LEARNED_BLENDS["hybrid_phase_gate_bucket_weights"][300]
    assert ten_minute_offset == module.MULTIRES_ROLLOUT_LEARNED_BLENDS["hybrid_phase_gate_bucket_weights"][600]
    assert unmatched_offset == module.MULTIRES_ROLLOUT_LEARNED_BLENDS["hybrid_phase_gate_non_aligned_weight"]


def test_build_phase_bucket_policy_candidates_uses_bucket_champions():
    """Derived phase-bucket policies should route each bucket to its measured champion candidate."""
    module = _load_recursive_rollout_module()
    by_origin = pd.DataFrame(
        [
            {
                "origin_timestamp": "2025-12-26T00:00:00",
                "candidate_label": "anchored_workday",
                "candidate_type": "baseline",
                "source_model_label": "anchored_workday",
                "target_mode": "baseline",
                "endpoint_abs_error": 210.0,
                "endpoint_sq_error": 210.0**2,
                "endpoint_actual_abs": 1000.0,
                "path_mae": 200.0,
                "path_rmse": 200.0,
                "path_abs_error_sum": 600.0,
                "path_actual_abs_sum": 3000.0,
                "phase_mean_abs_error": 180.0,
                "phase_mean_sq_error": 180.0**2,
                "phase_mean_actual_abs": 1000.0,
                "coverage": 1.0,
                "n_eval": 3.0,
            },
            {
                "origin_timestamp": "2025-12-26T00:00:00",
                "candidate_label": "model::raw",
                "candidate_type": "learned",
                "source_model_label": "model",
                "target_mode": "raw",
                "endpoint_abs_error": 400.0,
                "endpoint_sq_error": 400.0**2,
                "endpoint_actual_abs": 1000.0,
                "path_mae": 450.0,
                "path_rmse": 450.0,
                "path_abs_error_sum": 1350.0,
                "path_actual_abs_sum": 3000.0,
                "phase_mean_abs_error": 420.0,
                "phase_mean_sq_error": 420.0**2,
                "phase_mean_actual_abs": 1000.0,
                "coverage": 1.0,
                "n_eval": 3.0,
            },
            {
                "origin_timestamp": "2025-12-26T00:05:00",
                "candidate_label": "model::persistence_residual_blend_e40",
                "candidate_type": "learned",
                "source_model_label": "model",
                "target_mode": "persistence_residual_blend",
                "endpoint_abs_error": 310.0,
                "endpoint_sq_error": 310.0**2,
                "endpoint_actual_abs": 1000.0,
                "path_mae": 220.0,
                "path_rmse": 220.0,
                "path_abs_error_sum": 660.0,
                "path_actual_abs_sum": 3000.0,
                "phase_mean_abs_error": 160.0,
                "phase_mean_sq_error": 160.0**2,
                "phase_mean_actual_abs": 1000.0,
                "coverage": 1.0,
                "n_eval": 3.0,
            },
            {
                "origin_timestamp": "2025-12-26T00:05:00",
                "candidate_label": "anchored_workday",
                "candidate_type": "baseline",
                "source_model_label": "anchored_workday",
                "target_mode": "baseline",
                "endpoint_abs_error": 330.0,
                "endpoint_sq_error": 330.0**2,
                "endpoint_actual_abs": 1000.0,
                "path_mae": 260.0,
                "path_rmse": 260.0,
                "path_abs_error_sum": 780.0,
                "path_actual_abs_sum": 3000.0,
                "phase_mean_abs_error": 190.0,
                "phase_mean_sq_error": 190.0**2,
                "phase_mean_actual_abs": 1000.0,
                "coverage": 1.0,
                "n_eval": 3.0,
            },
            {
                "origin_timestamp": "2025-12-26T00:10:00",
                "candidate_label": "persistence",
                "candidate_type": "baseline",
                "source_model_label": "persistence",
                "target_mode": "baseline",
                "endpoint_abs_error": 170.0,
                "endpoint_sq_error": 170.0**2,
                "endpoint_actual_abs": 1000.0,
                "path_mae": 150.0,
                "path_rmse": 150.0,
                "path_abs_error_sum": 450.0,
                "path_actual_abs_sum": 3000.0,
                "phase_mean_abs_error": 120.0,
                "phase_mean_sq_error": 120.0**2,
                "phase_mean_actual_abs": 1000.0,
                "coverage": 1.0,
                "n_eval": 3.0,
            },
            {
                "origin_timestamp": "2025-12-26T00:10:00",
                "candidate_label": "model::hybrid_phase_gate",
                "candidate_type": "learned",
                "source_model_label": "model",
                "target_mode": "hybrid_phase_gate",
                "endpoint_abs_error": 160.0,
                "endpoint_sq_error": 160.0**2,
                "endpoint_actual_abs": 1000.0,
                "path_mae": 180.0,
                "path_rmse": 180.0,
                "path_abs_error_sum": 540.0,
                "path_actual_abs_sum": 3000.0,
                "phase_mean_abs_error": 110.0,
                "phase_mean_sq_error": 110.0**2,
                "phase_mean_actual_abs": 1000.0,
                "coverage": 1.0,
                "n_eval": 3.0,
            },
        ]
    )

    derived_rows, metadata = module._build_phase_bucket_policy_candidates(
        by_origin,
        model_label="model",
        horizon_minutes=15,
    )

    assert set(derived_rows["candidate_label"]) == {
        "model::phase_bucket_endpoint_policy",
        "model::phase_bucket_path_policy",
        "model::phase_bucket_phase_policy",
    }
    path_rows = derived_rows.loc[
        derived_rows["candidate_label"].astype("string").eq("model::phase_bucket_path_policy")
    ].sort_values("origin_timestamp", kind="stable")
    assert list(path_rows["policy_source_candidate"]) == [
        "anchored_workday",
        "model::persistence_residual_blend_e40",
        "persistence",
    ]
    phase_rows = derived_rows.loc[
        derived_rows["candidate_label"].astype("string").eq("model::phase_bucket_phase_policy")
    ].sort_values("origin_timestamp", kind="stable")
    assert list(phase_rows["policy_source_candidate"]) == [
        "anchored_workday",
        "model::persistence_residual_blend_e40",
        "model::hybrid_phase_gate",
    ]
    assert all(derived_rows["candidate_type"].astype("string").eq("learned"))
    assert len(metadata) == 3
    assert metadata[1]["candidate_label"] == "model::phase_bucket_path_policy"
    assert metadata[1]["phase_bucket_mapping"] == {
        0: "anchored_workday",
        300: "model::persistence_residual_blend_e40",
        600: "persistence",
    }


def test_anchored_workday_path_preserves_origin_level():
    """Anchored workday paths should keep the origin level while preserving profile shape deltas."""
    base = _synthetic_base()
    frame = build_causal_feature_frame(base, "5min")
    profile = build_workday_profile(frame)
    origin_position = 24
    origin_timestamp = pd.Timestamp(base.iloc[origin_position]["timestamp"])
    history = base.iloc[: origin_position + 1].set_index("timestamp")["avg_load"].astype(float)
    day_lookup = {timestamp.normalize(): "full" for timestamp in base["timestamp"]}

    anchored = anchored_workday_path(
        profile,
        history=history,
        origin_timestamp=origin_timestamp,
        horizon_steps=3,
        resolution="5min",
        day_class_lookup=day_lookup,
    )
    raw_profile = avg_workday_path(
        profile,
        origin_timestamp=origin_timestamp,
        horizon_steps=3,
        resolution="5min",
        day_class_lookup=day_lookup,
    )

    assert len(anchored) == 3
    assert np.isfinite(anchored["y_pred"]).all()
    assert np.allclose(
        np.diff(anchored["y_pred"].to_numpy(dtype=float)),
        np.diff(raw_profile["y_pred"].to_numpy(dtype=float)),
    )


def test_blend_candidate_paths_interpolates_between_inputs():
    """Blended candidate paths should interpolate smoothly from the primary to secondary path."""
    timestamps = pd.date_range("2025-12-01 00:05:00", periods=3, freq="5min")
    persistence = pd.DataFrame({"timestamp": timestamps, "y_pred": [100.0, 100.0, 100.0]})
    profile = pd.DataFrame({"timestamp": timestamps, "y_pred": [100.0, 110.0, 130.0]})

    blended = blend_candidate_paths(
        persistence,
        profile,
        primary_weight_start=1.0,
        primary_weight_end=0.5,
    )

    assert list(blended["timestamp"]) == list(timestamps)
    assert blended.iloc[0]["y_pred"] == pytest.approx(100.0)
    assert blended.iloc[-1]["y_pred"] == pytest.approx(115.0)


def test_select_rollout_origins_supports_uniform_midnight_billing_and_phase_policies():
    """Origin selection should honor the supported sampling policies used by Stage-7."""
    module = _load_recursive_rollout_module()
    base = _synthetic_base(days=4, freq="10min")
    module.SPLIT_DAY_RANGES = {"train": (1, 2), "validate": (3, 3), "test": (3, 4)}

    uniform = module._select_rollout_origins(
        base,
        horizon_steps=6,
        max_origins=4,
        origin_policy="uniform",
    )
    midnight = module._select_rollout_origins(
        base,
        horizon_steps=6,
        max_origins=4,
        origin_policy="midnight",
    )
    billing_aligned = module._select_rollout_origins(
        base,
        horizon_steps=6,
        max_origins=4,
        origin_policy="billing_aligned",
    )
    phase_balanced = module._select_rollout_origins(
        base,
        horizon_steps=6,
        max_origins=4,
        origin_policy="phase_balanced",
    )

    assert len(uniform) == 4
    assert len(midnight) >= 1
    assert len(billing_aligned) >= 1
    assert len(phase_balanced) == 4
    assert uniform[0] < uniform[-1]
    assert all(base.iloc[idx]["timestamp"].minute == 0 and base.iloc[idx]["timestamp"].hour == 0 for idx in midnight)
    assert all(base.iloc[idx]["timestamp"].minute % 15 == 0 for idx in billing_aligned)
    phase_buckets = {
        (base.iloc[idx]["timestamp"].minute * 60 + base.iloc[idx]["timestamp"].second) % (15 * 60)
        for idx in phase_balanced
    }
    assert len(phase_buckets) >= 2


def test_phase_balanced_origins_sample_across_the_full_quarter_hour_cycle():
    """Phase-balanced origin sampling should cover the entire 15-minute billing cycle."""
    module = _load_recursive_rollout_module()
    timestamps = pd.date_range("2025-12-01 00:00:00", periods=4 * 24 * 60, freq="1min")
    base = pd.DataFrame(
        {
            "timestamp": timestamps,
            "avg_load": np.linspace(100.0, 200.0, num=len(timestamps), dtype=float),
            "day_class": ["full"] * len(timestamps),
            "day_idx": np.repeat(np.arange(1, 5), 24 * 60),
        }
    )
    module.SPLIT_DAY_RANGES = {"train": (1, 2), "validate": (3, 3), "test": (3, 4)}

    selected = module._select_rollout_origins(
        base,
        horizon_steps=15,
        max_origins=8,
        origin_policy="phase_balanced",
    )

    phase_buckets = sorted(
        {
            (int(base.iloc[idx]["timestamp"].minute) * 60 + int(base.iloc[idx]["timestamp"].second)) % (15 * 60)
            for idx in selected
        }
    )

    assert len(selected) == 8
    assert len(phase_buckets) == 8
    assert phase_buckets[0] == 0
    assert phase_buckets[-1] == 14 * 60


def test_resolve_explicit_rollout_origins_maps_shared_timestamps_back_to_positions():
    """Explicit shared timestamps should resolve back to eligible rollout positions deterministically."""
    module = _load_recursive_rollout_module()
    base = _synthetic_base(days=4, freq="5min")
    module.SPLIT_DAY_RANGES = {"train": (1, 2), "validate": (3, 3), "test": (3, 4)}
    eligible_positions = module._select_rollout_origins(
        base,
        horizon_steps=12,
        max_origins=len(base),
        origin_policy="uniform",
    )
    requested_timestamps = [
        pd.Timestamp(base.iloc[eligible_positions[1]]["timestamp"]),
        pd.Timestamp(base.iloc[eligible_positions[4]]["timestamp"]),
        pd.Timestamp(base.iloc[eligible_positions[7]]["timestamp"]),
    ]

    resolved = module._resolve_explicit_rollout_origins(
        base,
        horizon_steps=12,
        origin_policy="uniform",
        origin_timestamps=requested_timestamps,
    )

    assert resolved == [eligible_positions[1], eligible_positions[4], eligible_positions[7]]


def test_resolve_explicit_rollout_origins_rejects_missing_timestamps():
    """Explicit timestamp resolution should fail fast when a requested origin is not representable."""
    module = _load_recursive_rollout_module()
    base = _synthetic_base(days=4, freq="5min")
    module.SPLIT_DAY_RANGES = {"train": (1, 2), "validate": (3, 3), "test": (3, 4)}

    with pytest.raises(ValueError, match="Explicit rollout origins are not representable"):
        module._resolve_explicit_rollout_origins(
            base,
            horizon_steps=12,
            origin_policy="uniform",
            origin_timestamps=[pd.Timestamp("2025-12-31T00:00:00")],
        )


def test_resolve_explicit_rollout_origins_ignores_nominal_origin_policy_filter():
    """Explicit shared timestamps should remain valid even when the candidate's policy is narrower."""
    module = _load_recursive_rollout_module()
    base = _synthetic_base(days=4, freq="5min")
    module.SPLIT_DAY_RANGES = {"train": (1, 2), "validate": (3, 3), "test": (3, 4)}
    eligible_positions = module._select_rollout_origins(
        base,
        horizon_steps=12,
        max_origins=len(base),
        origin_policy="uniform",
    )
    requested_timestamp = pd.Timestamp(base.iloc[eligible_positions[5]]["timestamp"])

    resolved = module._resolve_explicit_rollout_origins(
        base,
        horizon_steps=12,
        origin_policy="midnight",
        origin_timestamps=[requested_timestamp],
    )

    assert resolved == [eligible_positions[5]]


def test_resolve_rollout_feature_columns_adds_avg_workday_residual_support_to_minimal():
    """Minimal rollout features should expand to include the columns needed for workday residuals."""
    module = _load_recursive_rollout_module()
    columns = module._resolve_rollout_feature_columns("minimal", residual_baseline="avg_workday")

    assert "avg_workday_baseline" in columns
    assert "anchored_workday_baseline" in columns
    assert "profile_residual_lag_1" in columns
    assert "previous_day_residual" in columns


def test_resolve_rollout_feature_columns_adds_anchored_workday_residual_support():
    """Phase-aware minimal features should include the anchored-workday residual support columns."""
    module = _load_recursive_rollout_module()
    columns = module._resolve_rollout_feature_columns("minimal_phase", residual_baseline="anchored_workday")

    assert "anchored_workday_baseline" in columns
    assert "avg_workday_baseline" in columns
    assert "profile_residual_lag_1" in columns
    assert "previous_day_residual" in columns


def test_resolve_rollout_feature_columns_adds_hybrid_workday_residual_support():
    """Hybrid residuals should request both phase-profile context and the persistence anchor."""
    module = _load_recursive_rollout_module()
    columns = module._resolve_rollout_feature_columns("minimal", residual_baseline="hybrid_workday")

    assert "lag_1" in columns
    assert "anchored_workday_baseline" in columns
    assert "profile_residual_lag_1" in columns
    assert "previous_day_residual" in columns




def test_resolve_selection_context_uses_exact_horizon_match(tmp_path, monkeypatch):
    """Selection context should promote an exact Stage-6 horizon winner when one exists."""
    module = _load_recursive_rollout_module()
    latest_dir = tmp_path / "outputs" / "006_multires" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "use_case": "matched_horizon_30m",
                "winner_type": "learned_model",
                "winner_resolution": "5min",
                "winner_feature_set": "full",
                "winner_model_label": "hgb-frontier-lr010-leaf100",
                "winner_forecast_strategy": "recursive",
                "winner_horizon_minutes": 30,
                "decision_reason": "selected",
                "practical_gain_passed": True,
                "pareto_passed": True,
            },
            {
                "use_case": "matched_horizon_60m",
                "winner_type": "learned_model",
                "winner_resolution": "10min",
                "winner_feature_set": "curated",
                "winner_model_label": "hgb-balanced",
                "winner_forecast_strategy": "recursive",
                "winner_horizon_minutes": 60,
                "decision_reason": "selected",
                "practical_gain_passed": True,
                "pareto_passed": True,
            },
        ]
    ).to_csv(latest_dir / "selection_summary.csv", index=False)
    monkeypatch.setitem(module.PATHS, "outputs_multires_dir", tmp_path / "outputs" / "006_multires")
    monkeypatch.setitem(module.PATHS, "outputs_rollout_dir", tmp_path / "outputs" / "007_rollout")

    selection = module._resolve_selection_context(
        resolution=None,
        feature_set=None,
        model_label=None,
        requested_horizon_minutes=60,
        requested_origin_policy="uniform",
        selection_target="path_mae",
    )

    assert selection["resolution"] == "10min"
    assert selection["feature_set"] == "curated"
    assert selection["model_label"] == "hgb-balanced"
    assert selection["matched_stage6_horizon_minutes"] == 60
    assert selection["selection_source"].endswith("selection_summary.csv")


def test_resolve_selection_context_falls_back_when_no_exact_horizon_match(tmp_path, monkeypatch):
    """Selection context should fall back to config defaults when no exact upstream winner exists."""
    module = _load_recursive_rollout_module()
    latest_dir = tmp_path / "outputs" / "006_multires" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "use_case": "matched_horizon_30m",
                "winner_type": "learned_model",
                "winner_resolution": "5min",
                "winner_feature_set": "full",
                "winner_model_label": "hgb-frontier-lr010-leaf100",
                "winner_forecast_strategy": "recursive",
                "winner_horizon_minutes": 30,
                "decision_reason": "selected",
                "practical_gain_passed": True,
                "pareto_passed": True,
            }
        ]
    ).to_csv(latest_dir / "selection_summary.csv", index=False)
    monkeypatch.setitem(module.PATHS, "outputs_multires_dir", tmp_path / "outputs" / "006_multires")
    monkeypatch.setitem(module.PATHS, "outputs_rollout_dir", tmp_path / "outputs" / "007_rollout")

    selection = module._resolve_selection_context(
        resolution=None,
        feature_set=None,
        model_label=None,
        requested_horizon_minutes=1440,
        requested_origin_policy="uniform",
        selection_target="path_mae",
    )

    assert selection["resolution"] == module.canonical_resolution(module.MULTIRES_ROLLOUT["selected_resolution"])
    assert selection["feature_set"] == module.MULTIRES_ROLLOUT["feature_set"]
    assert selection["model_label"] == module.MULTIRES_ROLLOUT["model_label"]
    assert selection["selection_source"] == "multires.toml"
    assert selection["matched_stage6_horizon_minutes"] is None


def test_resolve_selection_context_prefers_winner_registry_over_latest_alias(tmp_path, monkeypatch):
    """Cross-run winner registries should outrank the mutable latest alias during auto-selection."""
    module = _load_recursive_rollout_module()
    outputs_dir = tmp_path / "outputs" / "006_multires"
    latest_dir = outputs_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "use_case": "matched_horizon_60m",
                "winner_type": "baseline_model",
                "winner_resolution": "10min",
                "winner_feature_set": "baseline",
                "winner_model_label": "avg_workday",
                "winner_forecast_strategy": "path_baseline",
                "winner_horizon_minutes": 60,
                "decision_reason": "selected",
                "practical_gain_passed": True,
                "pareto_passed": True,
            }
        ]
    ).to_csv(latest_dir / "selection_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_id": "20260307T133220706885Z",
                "generated_at_utc": "2026-03-07T13:32:20.706885+00:00",
                "mode": "candidate",
                "comparison_mode": "matched_horizon",
                "selection_summary_path": "outputs/006_multires/20260307T133220706885Z/selection_summary.csv",
                "use_case": "matched_horizon_120m",
                "winner_type": "learned_model",
                "winner_resolution": "5min",
                "winner_feature_set": "curated",
                "winner_model_label": "hgb-balanced",
                "winner_forecast_strategy": "recursive",
                "winner_horizon_minutes": 120,
                "decision_reason": "selected",
                "practical_gain_passed": True,
                "pareto_passed": True,
            }
        ]
    ).to_csv(outputs_dir / "winner_registry.csv", index=False)
    monkeypatch.setitem(module.PATHS, "outputs_multires_dir", outputs_dir)
    monkeypatch.setitem(module.PATHS, "outputs_rollout_dir", tmp_path / "outputs" / "007_rollout")

    selection = module._resolve_selection_context(
        resolution=None,
        feature_set=None,
        model_label=None,
        requested_horizon_minutes=120,
        requested_origin_policy="uniform",
        selection_target="path_mae",
    )

    assert selection["selection_source"].endswith("winner_registry.csv")
    assert selection["selection_run_id"] == "20260307T133220706885Z"
    assert selection["resolution"] == "5min"
    assert selection["feature_set"] == "curated"
    assert selection["model_label"] == "hgb-balanced"


def test_resolve_selection_context_uses_explicit_stage6_run_id(tmp_path, monkeypatch):
    """An explicit Stage-6 run id should pin selection to that run's saved summary."""
    module = _load_recursive_rollout_module()
    outputs_dir = tmp_path / "outputs" / "006_multires"
    run_id = "20260308T010203000000Z"
    run_dir = outputs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "use_case": "matched_horizon_60m",
                "winner_type": "learned_model",
                "winner_resolution": "5min",
                "winner_feature_set": "minimal",
                "winner_model_label": "ridge-medium",
                "winner_forecast_strategy": "recursive",
                "winner_horizon_minutes": 60,
                "decision_reason": "selected",
                "practical_gain_passed": True,
                "pareto_passed": True,
            }
        ]
    ).to_csv(run_dir / "selection_summary.csv", index=False)
    monkeypatch.setitem(module.PATHS, "outputs_multires_dir", outputs_dir)
    monkeypatch.setitem(module.PATHS, "outputs_rollout_dir", tmp_path / "outputs" / "007_rollout")

    selection = module._resolve_selection_context(
        resolution=None,
        feature_set=None,
        model_label=None,
        requested_horizon_minutes=60,
        requested_origin_policy="uniform",
        selection_target="path_mae",
        selection_run_id=run_id,
    )

    assert selection["selection_run_id"] == run_id
    assert selection["selection_source"].endswith("selection_summary.csv")
    assert selection["resolution"] == "5min"
    assert selection["feature_set"] == "minimal"
    assert selection["model_label"] == "ridge-medium"


def test_resolve_selection_context_ignores_direct_endpoint_winners_for_rollout(tmp_path, monkeypatch):
    """Rollout auto-selection should reject direct-endpoint winners when recursive replay is required."""
    module = _load_recursive_rollout_module()
    latest_dir = tmp_path / "outputs" / "006_multires" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "use_case": "matched_horizon_60m",
                "winner_type": "learned_model",
                "winner_resolution": "10min",
                "winner_feature_set": "full",
                "winner_model_label": "hgb-balanced",
                "winner_forecast_strategy": "direct_endpoint",
                "winner_horizon_minutes": 60,
                "decision_reason": "selected",
                "practical_gain_passed": True,
                "pareto_passed": True,
            }
        ]
    ).to_csv(latest_dir / "selection_summary.csv", index=False)
    monkeypatch.setitem(module.PATHS, "outputs_multires_dir", tmp_path / "outputs" / "006_multires")
    monkeypatch.setitem(module.PATHS, "outputs_rollout_dir", tmp_path / "outputs" / "007_rollout")

    selection = module._resolve_selection_context(
        resolution=None,
        feature_set=None,
        model_label=None,
        requested_horizon_minutes=60,
        requested_origin_policy="uniform",
        selection_target="path_mae",
    )

    assert selection["resolution"] == module.canonical_resolution(module.MULTIRES_ROLLOUT["selected_resolution"])
    assert selection["feature_set"] == module.MULTIRES_ROLLOUT["feature_set"]
    assert selection["model_label"] == module.MULTIRES_ROLLOUT["model_label"]
    assert selection["forecast_strategy"] == "recursive"
    assert selection["selection_source"] == "multires.toml"


def test_resolve_selection_context_ignores_baseline_winners_for_rollout(tmp_path, monkeypatch):
    """Rollout auto-selection should not blindly promote Stage-6 baseline winners as learned runs."""
    module = _load_recursive_rollout_module()
    latest_dir = tmp_path / "outputs" / "006_multires" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "use_case": "matched_horizon_120m",
                "winner_type": "baseline_model",
                "winner_resolution": "10min",
                "winner_feature_set": "baseline",
                "winner_model_label": "anchored_workday",
                "winner_forecast_strategy": "path_baseline",
                "winner_horizon_minutes": 120,
                "decision_reason": "selected",
                "practical_gain_passed": True,
                "pareto_passed": True,
            }
        ]
    ).to_csv(latest_dir / "selection_summary.csv", index=False)
    monkeypatch.setitem(module.PATHS, "outputs_multires_dir", tmp_path / "outputs" / "006_multires")
    monkeypatch.setitem(module.PATHS, "outputs_rollout_dir", tmp_path / "outputs" / "007_rollout")

    selection = module._resolve_selection_context(
        resolution=None,
        feature_set=None,
        model_label=None,
        requested_horizon_minutes=120,
        requested_origin_policy="uniform",
        selection_target="path_mae",
    )

    assert selection["selection_source"] == "multires.toml"
    assert selection["forecast_strategy"] == "recursive"


def test_resolve_selection_context_uses_rollout_registry_when_stage6_has_no_exact_match(tmp_path, monkeypatch):
    """Long-horizon selection should reuse saved rollout-registry evidence when Stage-6 cannot match."""
    module = _load_recursive_rollout_module()
    rollout_dir = tmp_path / "outputs" / "007_rollout"
    _write_rollout_history_run(
        rollout_dir,
        run_id="20260309T194339879991Z",
        resolution="5min",
        feature_set="minimal",
        model_label="hgb-frontier-lr010-leaf100",
        horizon_minutes=1440,
        origin_policy="uniform",
        selection_target="path_mae",
        learned_endpoint_mae=1165.424461,
        learned_path_mae=819.309448,
        persistence_endpoint_mae=1321.465669,
        persistence_path_mae=1023.300667,
        avg_workday_endpoint_mae=1108.193624,
        avg_workday_path_mae=872.270532,
    )
    monkeypatch.setitem(module.PATHS, "outputs_multires_dir", tmp_path / "outputs" / "006_multires")
    monkeypatch.setitem(module.PATHS, "outputs_rollout_dir", rollout_dir)

    selection = module._resolve_selection_context(
        resolution=None,
        feature_set=None,
        model_label=None,
        requested_horizon_minutes=1440,
        requested_origin_policy="uniform",
        selection_target="path_mae",
    )

    assert selection["selection_source"].endswith("rollout_registry.csv")
    assert selection["selection_policy"] == "stage7_rollout_registry"
    assert selection["selection_run_stage"] == "007_rollout"
    assert selection["selection_run_id"] == "20260309T194339879991Z"
    assert selection["matched_rollout_registry_horizon_minutes"] == 1440
    assert selection["resolution"] == "5min"
    assert selection["feature_set"] == "minimal"
    assert selection["model_label"] == "hgb-frontier-lr010-leaf100"


def test_resolve_selection_context_prefers_challenger_sweep_registry_for_matched_rollout_objective(
    tmp_path, monkeypatch
):
    """Objective-matched challenger sweeps should outrank generic upstream winners for rollout replay."""
    module = _load_recursive_rollout_module()
    multires_dir = tmp_path / "outputs" / "006_multires"
    multires_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run_id": "20260310T022740172522Z",
                "generated_at_utc": "2026-03-10T02:27:40.172522+00:00",
                "winner_type": "learned_model",
                "winner_resolution": "1min",
                "winner_feature_set": "minimal",
                "winner_model_label": "hgb-balanced",
                "winner_forecast_strategy": "recursive",
                "winner_horizon_minutes": 60,
                "practical_gain_passed": True,
                "pareto_passed": True,
            }
        ]
    ).to_csv(multires_dir / "winner_registry.csv", index=False)

    rollout_dir = tmp_path / "outputs" / "007_rollout"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "sweep_run_id": "20260311T011357499505Z",
                "generated_at_utc": "2026-03-11T01:17:28.401607+00:00",
                "requested_horizon_minutes": 60,
                "selection_target": "next_lock_mae",
                "origin_selection_scope": "shared_timestamp_intersection",
                "shared_origin_count": 8,
                "recommended_origin_policy": "phase_balanced",
                "recommended_candidate_label": "cross_candidate_portfolio::phase_bucket_next_lock_policy",
                "recommended_resolution": "mixed",
                "recommended_feature_set": "portfolio",
                "recommended_model_label": "cross_candidate_portfolio",
                "recommended_target_mode": "phase_bucket_next_lock_policy",
                "recommended_source_type": "cross_candidate_phase_bucket_portfolio",
                "recommended_run_id": "20260311T011357499505Z",
                "recommended_run_path": "outputs/007_rollout/commercial_facility/challenger_sweeps/20260311T011357499505Z",
                "recommended_metric_value": 253.104260,
                "recommended_metric_pct": 15.969845,
                "endpoint_mae": 751.696155,
                "path_mae": 496.893660,
                "next_lock_mae": 253.104260,
                "profile_shape_mae": 256.446567,
                "mean_coverage": 1.0,
                "origin_n": 8,
                "beats_persistence_next_lock": True,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_path": False,
                "beats_best_baseline_path": False,
                "beats_persistence_profile_shape": False,
                "beats_best_baseline_profile_shape": False,
                "recommended_candidate_path": "outputs/007_rollout/commercial_facility/challenger_sweeps/20260311T011357499505Z/recommended_candidate.json",
                "sweep_path": "outputs/007_rollout/commercial_facility/challenger_sweeps/20260311T011357499505Z",
            }
        ]
    ).to_csv(rollout_dir / "challenger_sweep_registry.csv", index=False)
    monkeypatch.setitem(module.PATHS, "outputs_multires_dir", multires_dir)
    monkeypatch.setitem(module.PATHS, "outputs_rollout_dir", rollout_dir)

    selection = module._resolve_selection_context(
        resolution=None,
        feature_set=None,
        model_label=None,
        requested_horizon_minutes=60,
        requested_origin_policy="phase_balanced",
        selection_target="next_lock_mae",
    )

    assert selection["selection_policy"] == "stage7_challenger_sweep_registry"
    assert selection["selection_run_stage"] == "007_rollout_challenger_sweep"
    assert selection["selection_run_id"] == "20260311T011357499505Z"
    assert selection["resolution"] == "mixed"
    assert selection["feature_set"] == "portfolio"
    assert selection["model_label"] == "cross_candidate_portfolio"
    assert selection["portfolio_candidate_label"] == "cross_candidate_portfolio::phase_bucket_next_lock_policy"
    assert selection["portfolio_policy_candidates_path"].endswith("portfolio_policy_candidates.json")


def test_rollout_registry_prefers_matching_origin_policy_and_target(tmp_path, monkeypatch):
    """Rollout registry reuse should prefer runs that match both origin policy and objective."""
    module = _load_recursive_rollout_module()
    rollout_dir = tmp_path / "outputs" / "007_rollout"
    _write_rollout_history_run(
        rollout_dir,
        run_id="20260309T100000000000Z",
        resolution="10min",
        feature_set="curated",
        model_label="ridge-medium",
        horizon_minutes=1440,
        origin_policy="midnight",
        selection_target="endpoint_mae",
        learned_endpoint_mae=400.0,
        learned_path_mae=1400.0,
        persistence_endpoint_mae=500.0,
        persistence_path_mae=1200.0,
        avg_workday_endpoint_mae=450.0,
        avg_workday_path_mae=1000.0,
        origins_per_run=2,
    )
    _write_rollout_history_run(
        rollout_dir,
        run_id="20260309T194339879991Z",
        resolution="5min",
        feature_set="minimal",
        model_label="hgb-frontier-lr010-leaf100",
        horizon_minutes=1440,
        origin_policy="uniform",
        selection_target="path_mae",
        learned_endpoint_mae=1165.424461,
        learned_path_mae=819.309448,
        persistence_endpoint_mae=1321.465669,
        persistence_path_mae=1023.300667,
        avg_workday_endpoint_mae=1108.193624,
        avg_workday_path_mae=872.270532,
        origins_per_run=8,
    )
    monkeypatch.setitem(module.PATHS, "outputs_multires_dir", tmp_path / "outputs" / "006_multires")
    monkeypatch.setitem(module.PATHS, "outputs_rollout_dir", rollout_dir)

    selection = module._resolve_selection_context(
        resolution=None,
        feature_set=None,
        model_label=None,
        requested_horizon_minutes=1440,
        requested_origin_policy="uniform",
        selection_target="path_mae",
    )

    assert selection["selection_policy"] == "stage7_rollout_registry"
    assert selection["selection_run_id"] == "20260309T194339879991Z"
    assert selection["resolution"] == "5min"
    assert selection["model_label"] == "hgb-frontier-lr010-leaf100"


def test_rollout_registry_supports_phase_mean_selection_target(tmp_path, monkeypatch):
    """Rollout registry selection should support phase-average objectives alongside path and endpoint."""
    module = _load_recursive_rollout_module()
    rollout_dir = tmp_path / "outputs" / "007_rollout"
    _write_rollout_history_run(
        rollout_dir,
        run_id="20260309T010000000000Z",
        resolution="5min",
        feature_set="minimal",
        model_label="hgb-balanced",
        horizon_minutes=15,
        origin_policy="billing_aligned",
        selection_target="phase_mean_mae",
        learned_endpoint_mae=620.0,
        learned_path_mae=460.0,
        persistence_endpoint_mae=600.0,
        persistence_path_mae=470.0,
        avg_workday_endpoint_mae=650.0,
        avg_workday_path_mae=480.0,
        learned_phase_mean_mae=210.0,
        persistence_phase_mean_mae=230.0,
        avg_workday_phase_mean_mae=240.0,
        origins_per_run=8,
    )
    _write_rollout_history_run(
        rollout_dir,
        run_id="20260309T020000000000Z",
        resolution="1min",
        feature_set="minimal",
        model_label="ridge-strong",
        horizon_minutes=15,
        origin_policy="uniform",
        selection_target="phase_mean_mae",
        learned_endpoint_mae=580.0,
        learned_path_mae=430.0,
        persistence_endpoint_mae=600.0,
        persistence_path_mae=470.0,
        avg_workday_endpoint_mae=650.0,
        avg_workday_path_mae=480.0,
        learned_phase_mean_mae=205.0,
        persistence_phase_mean_mae=230.0,
        avg_workday_phase_mean_mae=240.0,
        origins_per_run=8,
    )
    monkeypatch.setitem(module.PATHS, "outputs_multires_dir", tmp_path / "outputs" / "006_multires")
    monkeypatch.setitem(module.PATHS, "outputs_rollout_dir", rollout_dir)

    selection = module._resolve_selection_context(
        resolution=None,
        feature_set=None,
        model_label=None,
        requested_horizon_minutes=15,
        requested_origin_policy="billing_aligned",
        selection_target="phase_mean_mae",
    )

    assert selection["selection_policy"] == "stage7_rollout_registry"
    assert selection["selection_run_id"] == "20260309T010000000000Z"
    assert selection["resolution"] == "5min"
    assert selection["model_label"] == "hgb-balanced"
