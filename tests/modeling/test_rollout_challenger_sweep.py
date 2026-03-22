"""Unit tests for rollout challenger sweep selection helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    """Load the challenger sweep script as a module for direct helper testing."""
    path = Path("scripts/modeling/rollout_challenger_sweep.py").resolve()
    spec = importlib.util.spec_from_file_location("test_rollout_challenger_sweep_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finalize_candidate_plan_deduplicates_and_preserves_config_default():
    """Keep the highest-priority duplicate candidate while retaining the config fallback row."""
    module = _load_module()
    plan = module._finalize_candidate_plan(
        [
            {
                "source_priority": 1,
                "source_stage": "007_rollout",
                "source_type": "rollout_registry",
                "source_rank": 1,
                "source_run_id": "run-a",
                "source_selection_target": "path_mae",
                "source_horizon_minutes": 1440,
                "source_metric_value": 800.0,
                "source_metric_pct": 7.5,
                "source_metric_name": "path_mae",
                "source_path": "outputs/007_rollout/rollout_registry.csv",
                "resolution": "5min",
                "feature_set": "minimal",
                "model_label": "hgb-frontier-lr010-leaf100",
                "reason": "registry winner",
            },
            {
                "source_priority": 2,
                "source_stage": "006_multires",
                "source_type": "winner_registry",
                "source_rank": 1,
                "source_run_id": "run-b",
                "source_selection_target": pd.NA,
                "source_horizon_minutes": 120,
                "source_metric_value": pd.NA,
                "source_metric_pct": pd.NA,
                "source_metric_name": "matched_horizon_selection",
                "source_path": "outputs/006_multires/winner_registry.csv",
                "resolution": "5min",
                "feature_set": "minimal",
                "model_label": "hgb-frontier-lr010-leaf100",
                "reason": "duplicate candidate",
            },
            {
                "source_priority": 3,
                "source_stage": "multires.toml",
                "source_type": "config_default",
                "source_rank": 1,
                "source_run_id": pd.NA,
                "source_selection_target": "path_mae",
                "source_horizon_minutes": 1440,
                "source_metric_value": pd.NA,
                "source_metric_pct": pd.NA,
                "source_metric_name": "path_mae",
                "source_path": "config/multires.toml",
                "resolution": "10min",
                "feature_set": "full_stable",
                "model_label": "hgb-frontier-lr010-l2001",
                "reason": "default fallback",
            },
        ],
        max_candidates=2,
        preserve_default=True,
    )

    assert len(plan) == 2
    assert plan["source_type"].tolist() == ["rollout_registry", "config_default"]
    assert plan["candidate_rank"].tolist() == [1, 2]


def test_requested_origin_policies_for_sweep_respects_explicit_request():
    """Return only the caller-requested origin policy when the sweep is explicitly scoped."""
    module = _load_module()

    requested = module._requested_origin_policies_for_sweep("billing_aligned")

    assert requested == ["billing_aligned"]


def test_select_recommended_candidate_prefers_baseline_beating_objective_winner():
    """Prefer a candidate that clears the baseline gate over a numerically smaller unsupported row."""
    module = _load_module()
    results = pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "candidate_label": "5min/minimal/a",
                "selection_metric_value": 900.0,
                "endpoint_mae": 1200.0,
                "path_mae": 900.0,
                "mean_coverage": 1.0,
                "beats_best_baseline_endpoint": False,
                "beats_best_baseline_path": True,
                "beats_persistence_endpoint": False,
                "beats_persistence_path": True,
            },
            {
                "candidate_rank": 2,
                "candidate_label": "5min/minimal/b",
                "selection_metric_value": 890.0,
                "endpoint_mae": 1300.0,
                "path_mae": 890.0,
                "mean_coverage": 0.98,
                "beats_best_baseline_endpoint": False,
                "beats_best_baseline_path": False,
                "beats_persistence_endpoint": False,
                "beats_persistence_path": False,
            },
        ]
    )

    recommended = module._select_recommended_candidate(
        results,
        selection_target="path_mae",
        requested_origin_policy="uniform",
        recommendation_origin_scope="all",
    )

    assert recommended["candidate_label"] == "5min/minimal/a"


def test_select_recommended_candidate_prefers_requested_origin_policy_when_configured():
    """Restrict recommendation to the requested origin-policy slice when configured to do so."""
    module = _load_module()
    results = pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "candidate_label": "uniform_candidate",
                "requested_origin_policy": "uniform",
                "selection_metric_value": 520.0,
                "endpoint_mae": 700.0,
                "path_mae": 520.0,
                "mean_coverage": 1.0,
                "beats_best_baseline_endpoint": False,
                "beats_best_baseline_path": False,
                "beats_persistence_endpoint": False,
                "beats_persistence_path": False,
            },
            {
                "candidate_rank": 2,
                "candidate_label": "midnight_candidate",
                "requested_origin_policy": "midnight",
                "selection_metric_value": 410.0,
                "endpoint_mae": 650.0,
                "path_mae": 410.0,
                "mean_coverage": 1.0,
                "beats_best_baseline_endpoint": False,
                "beats_best_baseline_path": False,
                "beats_persistence_endpoint": False,
                "beats_persistence_path": False,
            },
        ]
    )

    recommended = module._select_recommended_candidate(
        results,
        selection_target="path_mae",
        requested_origin_policy="uniform",
        recommendation_origin_scope="requested_only",
    )

    assert recommended["candidate_label"] == "uniform_candidate"


def test_select_recommended_candidate_supports_phase_objective():
    """Evaluate phase-average objectives with the phase-specific baseline-beat flags."""
    module = _load_module()
    results = pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "candidate_label": "billing_candidate",
                "requested_origin_policy": "billing_aligned",
                "endpoint_mae": 620.0,
                "path_mae": 440.0,
                "phase_mean_mae": 205.0,
                "mean_coverage": 1.0,
                "beats_best_baseline_endpoint": False,
                "beats_best_baseline_path": False,
                "beats_best_baseline_phase": True,
                "beats_persistence_endpoint": False,
                "beats_persistence_path": True,
                "beats_persistence_phase": True,
            },
            {
                "candidate_rank": 2,
                "candidate_label": "uniform_candidate",
                "requested_origin_policy": "uniform",
                "endpoint_mae": 590.0,
                "path_mae": 430.0,
                "phase_mean_mae": 215.0,
                "mean_coverage": 1.0,
                "beats_best_baseline_endpoint": True,
                "beats_best_baseline_path": True,
                "beats_best_baseline_phase": False,
                "beats_persistence_endpoint": True,
                "beats_persistence_path": True,
                "beats_persistence_phase": True,
            },
        ]
    )

    recommended = module._select_recommended_candidate(
        results,
        selection_target="phase_mean_mae",
        requested_origin_policy="billing_aligned",
        recommendation_origin_scope="requested_only",
    )

    assert recommended["candidate_label"] == "billing_candidate"


def test_select_recommended_candidate_supports_next_lock_objective():
    """Evaluate next-lock objectives with the corresponding short-horizon support flags."""
    module = _load_module()
    results = pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "candidate_label": "phase_balanced_candidate",
                "requested_origin_policy": "phase_balanced",
                "endpoint_mae": 620.0,
                "path_mae": 430.0,
                "next_lock_mae": 180.0,
                "mean_coverage": 1.0,
                "beats_best_baseline_endpoint": False,
                "beats_best_baseline_path": True,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_endpoint": False,
                "beats_persistence_path": True,
                "beats_persistence_next_lock": True,
            },
            {
                "candidate_rank": 2,
                "candidate_label": "uniform_candidate",
                "requested_origin_policy": "uniform",
                "endpoint_mae": 600.0,
                "path_mae": 420.0,
                "next_lock_mae": 175.0,
                "mean_coverage": 1.0,
                "beats_best_baseline_endpoint": True,
                "beats_best_baseline_path": True,
                "beats_best_baseline_next_lock": False,
                "beats_persistence_endpoint": True,
                "beats_persistence_path": True,
                "beats_persistence_next_lock": True,
            },
        ]
    )

    recommended = module._select_recommended_candidate(
        results,
        selection_target="next_lock_mae",
        requested_origin_policy="phase_balanced",
        recommendation_origin_scope="requested_only",
    )

    assert recommended["candidate_label"] == "phase_balanced_candidate"


def test_select_challenger_sweep_registry_candidate_prefers_shared_origin_sweeps():
    """Prefer challenger sweeps that used shared-origin intersections over legacy incomparable runs."""
    module = _load_module()
    registry = pd.DataFrame(
        [
            {
                "sweep_run_id": "legacy-run",
                "requested_horizon_minutes": 15,
                "selection_target": "next_lock_mae",
                "recommended_origin_policy": "phase_balanced",
                "origin_selection_scope": "",
                "shared_origin_count": 0,
                "recommended_metric_value": 266.0,
                "next_lock_mae": 266.0,
                "path_mae": 430.0,
                "origin_n": 8,
                "mean_coverage": 1.0,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_next_lock": True,
            },
            {
                "sweep_run_id": "shared-run",
                "requested_horizon_minutes": 15,
                "selection_target": "next_lock_mae",
                "recommended_origin_policy": "phase_balanced",
                "origin_selection_scope": "shared_timestamp_intersection",
                "shared_origin_count": 8,
                "recommended_metric_value": 294.0,
                "next_lock_mae": 294.0,
                "path_mae": 410.0,
                "origin_n": 8,
                "mean_coverage": 1.0,
                "beats_best_baseline_next_lock": True,
                "beats_persistence_next_lock": True,
            },
        ]
    )

    recommended = module._select_challenger_sweep_registry_candidate(
        registry,
        requested_horizon_minutes=15,
        requested_origin_policy="phase_balanced",
        selection_target="next_lock_mae",
    )

    assert recommended is not None
    assert recommended["sweep_run_id"] == "shared-run"


def test_candidate_run_summary_prefers_best_learned_variant_from_rollout_run():
    """Summarize a rollout run using the strongest learned variant for the requested objective."""
    module = _load_module()
    result = {
        "run_dir": Path("outputs/007_rollout/commercial_facility/20260309T000000000000Z"),
        "metrics": pd.DataFrame(
            [
                {
                    "candidate_label": "hgb-balanced::raw",
                    "candidate_type": "learned",
                    "source_model_label": "hgb-balanced",
                    "target_mode": "raw",
                    "endpoint_mae": 1200.0,
                    "endpoint_mae_pct": 40.0,
                    "path_mae": 900.0,
                    "path_mae_pct": 35.0,
                    "phase_mean_mae": 850.0,
                    "phase_mean_mae_pct": 33.0,
                    "mean_coverage": 1.0,
                    "origin_n": 8,
                },
                {
                    "candidate_label": "hgb-balanced::avg_workday_residual",
                    "candidate_type": "learned",
                    "source_model_label": "hgb-balanced",
                    "target_mode": "avg_workday_residual",
                    "endpoint_mae": 950.0,
                    "endpoint_mae_pct": 32.0,
                    "path_mae": 700.0,
                    "path_mae_pct": 28.0,
                    "phase_mean_mae": 640.0,
                    "phase_mean_mae_pct": 26.0,
                    "mean_coverage": 1.0,
                    "origin_n": 8,
                },
                {
                    "candidate_label": "persistence",
                    "candidate_type": "baseline",
                    "source_model_label": "persistence",
                    "target_mode": "baseline",
                    "endpoint_mae": 1100.0,
                    "endpoint_mae_pct": 38.0,
                    "path_mae": 820.0,
                    "path_mae_pct": 31.0,
                    "phase_mean_mae": 780.0,
                    "phase_mean_mae_pct": 29.0,
                    "mean_coverage": 1.0,
                    "origin_n": 8,
                },
            ]
        ),
    }
    plan_row = pd.Series(
        {
            "candidate_rank": 1,
            "resolution": "10min",
            "feature_set": "minimal",
            "model_label": "hgb-balanced",
            "source_stage": "006_multires",
            "source_type": "winner_registry",
            "source_run_id": "run-a",
            "source_horizon_minutes": 1440,
            "requested_origin_policy": "uniform",
            "reason": "registry",
        }
    )

    summary = module._candidate_run_summary(
        result=result,
        plan_row=plan_row,
        selection_target="path_mae",
    )

    assert summary["candidate_label"] == "hgb-balanced::avg_workday_residual"
    assert summary["learned_target_mode"] == "avg_workday_residual"
    assert summary["phase_mean_mae"] == 640.0


def test_partition_representable_candidates_skips_nonrepresentable_resolutions():
    """Filter out candidate rows whose resolution cannot represent the requested rollout horizon."""
    module = _load_module()
    plan = pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "resolution": "10min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
            },
            {
                "candidate_rank": 2,
                "resolution": "5min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
            },
        ]
    )

    runnable, warnings = module._partition_representable_candidates(
        plan,
        requested_horizon_minutes=15,
    )

    assert runnable["resolution"].tolist() == ["5min"]
    assert warnings == ["skipped_non_representable_candidate:10min:15:minimal:hgb-balanced"]


def test_build_shared_origin_timestamps_intersects_candidate_resolutions(monkeypatch):
    """Intersect origin timestamps so all sweep candidates are evaluated on the same control points."""
    module = _load_module()
    candidate_plan = pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "resolution": "5min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
                "requested_origin_policy": "uniform",
            },
            {
                "candidate_rank": 2,
                "resolution": "10min",
                "feature_set": "full_stable",
                "model_label": "hgb-frontier-lr010-l2001",
                "requested_origin_policy": "uniform",
            },
        ]
    )

    def _fake_load_base_gold(resolution: str):
        if resolution == "5min":
            timestamps = pd.to_datetime(
                [
                    "2025-12-26T00:00:00",
                    "2025-12-26T00:05:00",
                    "2025-12-26T00:10:00",
                    "2025-12-26T00:15:00",
                    "2025-12-26T00:20:00",
                ]
            )
        else:
            timestamps = pd.to_datetime(
                [
                    "2025-12-26T00:00:00",
                    "2025-12-26T00:10:00",
                    "2025-12-26T00:20:00",
                    "2025-12-26T00:30:00",
                ]
            )
        return pd.DataFrame({"timestamp": timestamps})

    monkeypatch.setattr(module, "load_base_gold", _fake_load_base_gold)
    monkeypatch.setattr(
        module,
        "_select_rollout_origins",
        lambda base, *, horizon_steps, max_origins, origin_policy: list(range(len(base))),
    )

    shared, warnings = module._build_shared_origin_timestamps(
        candidate_plan=candidate_plan,
        requested_horizon_minutes=60,
        origins=2,
    )

    assert warnings == []
    assert shared["requested_origin_policy"].tolist() == ["uniform", "uniform"]
    assert shared["origin_timestamp"].tolist() == [
        "2025-12-26T00:00:00",
        "2025-12-26T00:20:00",
    ]


def test_build_cross_candidate_phase_bucket_policy_by_origin_switches_source_by_bucket():
    """Assemble a phase-bucket portfolio that borrows the best source candidate per bucket."""
    module = _load_module()

    def _source_frame(candidate_label: str, model_label: str, bucket_metrics: dict[str, tuple[float, float]]) -> pd.DataFrame:
        timestamps = pd.to_datetime(["2025-12-26T00:00:00", "2025-12-26T00:20:00"])
        rows = []
        for timestamp in timestamps:
            next_lock_mae, path_mae = bucket_metrics[timestamp.isoformat()]
            rows.append(
                {
                    "origin_timestamp": timestamp,
                    "candidate_label": candidate_label,
                    "candidate_type": "learned",
                    "source_model_label": model_label,
                    "target_mode": "raw",
                    "endpoint_abs_error": path_mae + 10.0,
                    "path_mae": path_mae,
                    "next_lock_mae": next_lock_mae,
                    "profile_shape_mae": next_lock_mae + 5.0,
                }
            )
            rows.append(
                {
                    "origin_timestamp": timestamp,
                    "candidate_label": "persistence",
                    "candidate_type": "baseline",
                    "source_model_label": "persistence",
                    "target_mode": "baseline",
                    "endpoint_abs_error": 100.0,
                    "path_mae": 100.0,
                    "next_lock_mae": 100.0,
                    "profile_shape_mae": 100.0,
                    "baseline_marker": candidate_label,
                }
            )
        return pd.DataFrame(rows)

    source_run_records = [
        {
            "summary": {
                "run_id": "run-a",
                "candidate_label": "hgb-balanced::raw",
                "resolution": "10min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
            },
            "result": {
                "by_origin": _source_frame(
                    "hgb-balanced::raw",
                    "hgb-balanced",
                    {
                        "2025-12-26T00:00:00": (60.0, 40.0),
                        "2025-12-26T00:20:00": (10.0, 70.0),
                    },
                )
            },
        },
        {
            "summary": {
                "run_id": "run-b",
                "candidate_label": "hgb-frontier::raw",
                "resolution": "5min",
                "feature_set": "minimal",
                "model_label": "hgb-frontier",
            },
            "result": {
                "by_origin": _source_frame(
                    "hgb-frontier::raw",
                    "hgb-frontier",
                    {
                        "2025-12-26T00:00:00": (20.0, 30.0),
                        "2025-12-26T00:20:00": (40.0, 20.0),
                    },
                )
            },
        },
    ]

    combined, metadata = module._build_cross_candidate_phase_bucket_policy_by_origin(
        source_run_records=source_run_records,
        selection_target="next_lock_mae",
        requested_origin_policy="phase_balanced",
    )

    assert metadata is not None
    assert metadata["phase_bucket_mapping"][0]["candidate_label"] == "hgb-frontier::raw"
    assert metadata["phase_bucket_mapping"][300]["candidate_label"] == "hgb-balanced::raw"

    selector_rows = combined.loc[
        combined["candidate_label"].astype("string").eq(
            "cross_candidate_portfolio::phase_bucket_next_lock_policy"
        )
    ].copy()
    selector_rows["origin_timestamp"] = pd.to_datetime(selector_rows["origin_timestamp"])
    selector_rows = selector_rows.sort_values("origin_timestamp").reset_index(drop=True)
    assert selector_rows["policy_source_candidate"].tolist() == [
        "hgb-frontier::raw",
        "hgb-balanced::raw",
    ]

    persistence_rows = combined.loc[combined["candidate_label"].astype("string").eq("persistence")].copy()
    persistence_rows["origin_timestamp"] = pd.to_datetime(persistence_rows["origin_timestamp"])
    persistence_rows = persistence_rows.sort_values("origin_timestamp").reset_index(drop=True)
    assert persistence_rows["baseline_marker"].tolist() == [
        "hgb-frontier::raw",
        "hgb-balanced::raw",
    ]


def test_build_horizon_policy_candidates_seeds_configured_policy_grid():
    """Seed candidate planning from the centralized horizon-policy configuration grid."""
    module = _load_module()

    candidates = module._build_horizon_policy_candidates(requested_horizon_minutes=240)
    candidate_rows = pd.DataFrame(candidates)

    assert not candidate_rows.empty
    assert set(candidate_rows["source_type"]) == {"horizon_policy"}
    assert candidate_rows["resolution"].isin(["1min", "5min", "10min"]).all()
    assert (
        candidate_rows["feature_set"].astype("string").eq("regime_profile").any()
    )
    assert (
        candidate_rows["model_label"].astype("string").eq("hgb-frontier-lr010-leaf100").any()
    )


def test_build_horizon_policy_candidates_skips_unavailable_optional_models(monkeypatch):
    """Filter optional policy-grid models out when they are unavailable locally."""
    module = _load_module()
    monkeypatch.setattr(
        module,
        "build_model_catalog",
        lambda: {
            "hgb-balanced": object(),
            "hgb-frontier-lr010-leaf100": object(),
        },
    )

    candidates = module._build_horizon_policy_candidates(requested_horizon_minutes=240)
    candidate_rows = pd.DataFrame(candidates)

    assert not candidate_rows.empty
    assert candidate_rows["model_label"].isin(["hgb-balanced", "hgb-frontier-lr010-leaf100"]).all()


def test_build_challenger_sweep_registry_snapshot_reads_recommended_artifacts(tmp_path):
    """Backfill the sweep registry from saved recommendation and candidate-result artifacts."""
    module = _load_module()
    output_root = tmp_path / "rollout"
    sweep_dir = output_root / "challenger_sweeps" / "20260310T020000000000Z"
    sweep_dir.mkdir(parents=True)

    (sweep_dir / "recommended_candidate.json").write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-03-10T02:20:00+00:00",
                "load_type": "commercial_facility",
                "artifact_namespace": "commercial_facility",
                "selection_target": "path_mae",
                "requested_horizon_minutes": 120,
                "recommended_origin_policy": "uniform",
                "recommended_candidate_label": "hgb-balanced::persistence_raw_blend_e35",
                "recommended_resolution": "10min",
                "recommended_feature_set": "minimal",
                "recommended_model_label": "hgb-balanced",
                "recommended_run_id": "20260310T020000111111Z",
                "recommended_run_path": "outputs/007_rollout/commercial_facility/20260310T020000111111Z",
                "recommended_metric_value": 607.220358,
                "recommended_metric_pct": 29.414596,
            }
        ),
        encoding="utf-8",
    )
    (sweep_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "20260310T020000000000Z",
                "stage": "007_rollout_challenger_sweep",
                "origin_selection_scope": "shared_timestamp_intersection",
                "shared_origin_count": 8,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "candidate_label": "hgb-balanced::persistence_raw_blend_e35",
                "resolution": "10min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
                "learned_target_mode": "raw",
                "requested_origin_policy": "uniform",
                "run_id": "20260310T020000111111Z",
                "run_path": "outputs/007_rollout/commercial_facility/20260310T020000111111Z",
                "endpoint_mae": 579.911901,
                "endpoint_mae_pct": 31.0,
                "path_mae": 607.220358,
                "path_mae_pct": 29.414596,
                "phase_mean_mae": 401.0,
                "phase_mean_mae_pct": 19.5,
                "selection_metric_value": 607.220358,
                "selection_metric_pct": 29.414596,
                "mean_coverage": 1.0,
                "origin_n": 8,
                "persistence_endpoint_mae": 506.931988,
                "persistence_endpoint_mae_pct": 30.0,
                "persistence_path_mae": 617.943901,
                "persistence_path_mae_pct": 29.934059,
                "persistence_phase_mean_mae": 430.0,
                "persistence_phase_mean_mae_pct": 21.0,
                "best_baseline_endpoint_label": "persistence",
                "best_baseline_endpoint_mae": 506.931988,
                "best_baseline_endpoint_mae_pct": 30.0,
                "best_baseline_path_label": "persistence",
                "best_baseline_path_mae": 617.943901,
                "best_baseline_path_mae_pct": 29.934059,
                "best_baseline_phase_label": "persistence",
                "best_baseline_phase_mae": 430.0,
                "best_baseline_phase_mae_pct": 21.0,
                "beats_persistence_endpoint": False,
                "beats_persistence_path": True,
                "beats_persistence_phase": True,
                "beats_best_baseline_endpoint": False,
                "beats_best_baseline_path": True,
                "beats_best_baseline_phase": True,
            }
        ]
    ).to_csv(sweep_dir / "candidate_results.csv", index=False)

    registry = module._build_challenger_sweep_registry_snapshot(output_root)

    assert len(registry) == 1
    row = registry.iloc[0]
    assert row["sweep_run_id"] == "20260310T020000000000Z"
    assert int(row["requested_horizon_minutes"]) == 120
    assert row["origin_selection_scope"] == "shared_timestamp_intersection"
    assert int(row["shared_origin_count"]) == 8
    assert row["recommended_origin_policy"] == "uniform"
    assert row["recommended_candidate_label"] == "hgb-balanced::persistence_raw_blend_e35"
    assert float(row["path_mae"]) == 607.220358
    assert float(row["phase_mean_mae"]) == 401.0
    assert bool(row["beats_persistence_path"]) is True


def test_select_challenger_sweep_registry_candidate_prefers_stronger_supported_evidence():
    """Prefer stronger challenger evidence with broader support over smaller but weaker legacy wins."""
    module = _load_module()
    registry = pd.DataFrame(
        [
            {
                "sweep_run_id": "older-2-origin-loss",
                "requested_horizon_minutes": 120,
                "selection_target": "path_mae",
                "recommended_origin_policy": "uniform",
                "recommended_metric_value": 416.063526,
                "endpoint_mae": 860.285326,
                "path_mae": 416.063526,
                "mean_coverage": 1.0,
                "origin_n": 2,
                "beats_persistence_endpoint": False,
                "beats_persistence_path": False,
                "beats_best_baseline_endpoint": False,
                "beats_best_baseline_path": False,
                "generated_at_utc": pd.Timestamp("2026-03-10T00:08:26Z"),
            },
            {
                "sweep_run_id": "newer-8-origin-win",
                "requested_horizon_minutes": 120,
                "selection_target": "path_mae",
                "recommended_origin_policy": "uniform",
                "recommended_metric_value": 607.220358,
                "endpoint_mae": 579.911901,
                "path_mae": 607.220358,
                "mean_coverage": 1.0,
                "origin_n": 8,
                "beats_persistence_endpoint": False,
                "beats_persistence_path": True,
                "beats_best_baseline_endpoint": False,
                "beats_best_baseline_path": True,
                "generated_at_utc": pd.Timestamp("2026-03-10T02:26:18Z"),
            },
        ]
    )

    selected = module._select_challenger_sweep_registry_candidate(
        registry,
        requested_horizon_minutes=120,
        requested_origin_policy="uniform",
        selection_target="path_mae",
    )

    assert selected is not None
    assert selected["sweep_run_id"] == "newer-8-origin-win"


def test_run_rollout_challenger_sweep_uses_selected_plus_baselines(tmp_path, monkeypatch):
    """Replay each challenger with only the selected label plus baseline support rows."""
    module = _load_module()
    plan = pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "source_priority": 1,
                "source_stage": "007_rollout",
                "source_type": "rollout_registry",
                "source_rank": 1,
                "source_run_id": "run-a",
                "source_selection_target": "path_mae",
                "source_horizon_minutes": 1440,
                "source_metric_value": 800.0,
                "source_metric_pct": 10.0,
                "source_metric_name": "path_mae",
                "source_path": "outputs/007_rollout/rollout_registry.csv",
                "resolution": "10min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
                "reason": "registry winner",
                "requested_origin_policy": "uniform",
            }
        ]
    )
    shared_origins = pd.DataFrame(
        [{"requested_origin_policy": "uniform", "origin_timestamp": "2026-03-10T00:00:00+00:00"}]
    )
    run_dir = tmp_path / "outputs" / "007_rollout" / "20260314T000000000000Z"
    run_dir.mkdir(parents=True, exist_ok=True)
    calls: list[str] = []

    def _stub_run_rollout_evaluation(**kwargs):
        calls.append(str(kwargs["candidate_scope"]))
        metrics = pd.DataFrame(
            [
                {
                    "candidate_label": "hgb-balanced::raw",
                    "candidate_type": "learned",
                    "target_mode": "raw",
                    "endpoint_mae": 100.0,
                    "endpoint_mae_pct": 10.0,
                    "path_mae": 90.0,
                    "path_mae_pct": 9.0,
                    "phase_mean_mae": 80.0,
                    "phase_mean_mae_pct": 8.0,
                    "next_lock_mae": 70.0,
                    "next_lock_mae_pct": 7.0,
                    "profile_shape_mae": 60.0,
                    "profile_shape_mae_pct": 6.0,
                    "energy_mae": 50.0,
                    "energy_mae_pct": 5.0,
                    "mean_coverage": 1.0,
                    "origin_n": 1,
                },
                {
                    "candidate_label": "persistence",
                    "candidate_type": "baseline",
                    "target_mode": "baseline",
                    "endpoint_mae": 110.0,
                    "endpoint_mae_pct": 11.0,
                    "path_mae": 100.0,
                    "path_mae_pct": 10.0,
                    "phase_mean_mae": 90.0,
                    "phase_mean_mae_pct": 9.0,
                    "next_lock_mae": 80.0,
                    "next_lock_mae_pct": 8.0,
                    "profile_shape_mae": 70.0,
                    "profile_shape_mae_pct": 7.0,
                    "energy_mae": 60.0,
                    "energy_mae_pct": 6.0,
                    "mean_coverage": 1.0,
                    "origin_n": 1,
                },
            ]
        )
        return {"run_dir": run_dir, "metrics": metrics}

    monkeypatch.setattr(module, "_build_challenger_plan", lambda **kwargs: plan.copy())
    monkeypatch.setattr(
        module,
        "_partition_representable_candidates",
        lambda candidate_plan, requested_horizon_minutes: (candidate_plan.copy(), []),
    )
    monkeypatch.setattr(
        module,
        "_build_shared_origin_timestamps",
        lambda **kwargs: (shared_origins.copy(), []),
    )
    monkeypatch.setattr(module, "run_rollout_evaluation", _stub_run_rollout_evaluation)
    monkeypatch.setattr(
        module,
        "_build_cross_candidate_phase_bucket_policies",
        lambda **kwargs: (pd.DataFrame(), [], pd.DataFrame()),
    )
    monkeypatch.setattr(
        module,
        "_build_challenger_sweep_registry_snapshot",
        lambda output_root: pd.DataFrame(columns=module.CHALLENGER_SWEEP_REGISTRY_COLUMNS),
    )
    monkeypatch.setattr(module, "_build_rollout_registry_snapshot", lambda output_root: pd.DataFrame())
    monkeypatch.setattr(module, "update_latest_alias", lambda *args, **kwargs: None)

    result = module.run_rollout_challenger_sweep(
        output_root=tmp_path / "outputs" / "007_rollout",
        horizon_minutes=1440,
        origins=1,
        origin_policy="uniform",
        selection_target="path_mae",
        max_candidates=1,
        refresh_rollout_registry=False,
        refresh_rollout_latest=False,
        refresh_sweep_latest=False,
    )

    assert calls == ["selected_plus_baselines"]
    manifest = json.loads((result["sweep_dir"] / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"


def test_run_rollout_challenger_sweep_writes_failed_manifest_on_exception(tmp_path, monkeypatch):
    """Keep failed sweep runs inspectable by persisting a failure manifest."""
    module = _load_module()
    plan = pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "source_priority": 1,
                "source_stage": "007_rollout",
                "source_type": "rollout_registry",
                "source_rank": 1,
                "source_run_id": "run-a",
                "source_selection_target": "path_mae",
                "source_horizon_minutes": 1440,
                "source_metric_value": 800.0,
                "source_metric_pct": 10.0,
                "source_metric_name": "path_mae",
                "source_path": "outputs/007_rollout/rollout_registry.csv",
                "resolution": "10min",
                "feature_set": "minimal",
                "model_label": "hgb-balanced",
                "reason": "registry winner",
                "requested_origin_policy": "uniform",
            }
        ]
    )
    shared_origins = pd.DataFrame(
        [{"requested_origin_policy": "uniform", "origin_timestamp": "2026-03-10T00:00:00+00:00"}]
    )

    monkeypatch.setattr(module, "_build_challenger_plan", lambda **kwargs: plan.copy())
    monkeypatch.setattr(
        module,
        "_partition_representable_candidates",
        lambda candidate_plan, requested_horizon_minutes: (candidate_plan.copy(), []),
    )
    monkeypatch.setattr(
        module,
        "_build_shared_origin_timestamps",
        lambda **kwargs: (shared_origins.copy(), []),
    )
    monkeypatch.setattr(
        module,
        "run_rollout_evaluation",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("simulated sweep failure")),
    )

    with pytest.raises(RuntimeError, match="simulated sweep failure"):
        module.run_rollout_challenger_sweep(
            output_root=tmp_path / "outputs" / "007_rollout",
            horizon_minutes=1440,
            origins=1,
            origin_policy="uniform",
            selection_target="path_mae",
            max_candidates=1,
            refresh_rollout_registry=False,
            refresh_rollout_latest=False,
            refresh_sweep_latest=False,
        )

    manifests = sorted((tmp_path / "outputs" / "007_rollout" / "challenger_sweeps").glob("*/run_manifest.json"))
    assert manifests
    manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "simulated sweep failure" in manifest["failure_reason"]
