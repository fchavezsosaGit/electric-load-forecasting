"""Integration tests for stage-6 and stage-7 multiresolution runners."""

from __future__ import annotations

import os
import subprocess
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scripts.config import DATASET


def _scoped_output_dir(tmp_path: Path, stage_dir: str) -> Path:
    """Return the load-type-scoped output directory used by the integration sandbox."""
    return tmp_path / "outputs" / stage_dir / DATASET["artifact_namespace"]


def _scoped_latest_dir(tmp_path: Path, stage_dir: str, alias: str = "latest") -> Path:
    """Return a scoped latest-style alias directory for a stage within the sandbox."""
    return _scoped_output_dir(tmp_path, stage_dir) / alias


def _normalize_rollout_output_root(output_root: Path) -> Path:
    """Normalize rollout roots so helpers accept either the stage root or the artifact namespace root."""
    if output_root.name == DATASET["artifact_namespace"]:
        return output_root
    return output_root / DATASET["artifact_namespace"]


def _write_gold(config_dir: Path, resolution: str, periods_per_day: int, freq: str) -> None:
    """Seed a minimal gold parquet at the requested resolution for CLI smoke and contract tests."""
    data_dir = config_dir.parent / "data" / "003_gold"
    data_dir.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range("2025-11-28 00:00:00", periods=31 * periods_per_day, freq=freq)
    signal = 100.0 + np.sin(np.arange(len(timestamps)) / 12.0) * 10.0
    day_idx = np.repeat(np.arange(1, 32), periods_per_day)
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "avg_load": signal.astype(float),
            "day_class": ["full"] * len(timestamps),
            "workday": [2] * len(timestamps),
            "year": timestamps.year,
            "quarter": timestamps.quarter,
            "month": timestamps.month,
            "day": timestamps.day,
            "day_of_week": ((timestamps.dayofweek + 1) % 7).astype(int),
            "hour": timestamps.hour,
            "season": [1] * len(timestamps),
            "time_of_day": [0] * len(timestamps),
            "hour_sin": np.sin(2 * np.pi * (timestamps.hour + timestamps.minute / 60.0 + timestamps.second / 3600.0) / 24.0),
            "hour_cos": np.cos(2 * np.pi * (timestamps.hour + timestamps.minute / 60.0 + timestamps.second / 3600.0) / 24.0),
            "dow_sin": np.sin(2 * np.pi * (((timestamps.dayofweek + 1) % 7) + (timestamps.hour + timestamps.minute / 60.0 + timestamps.second / 3600.0) / 24.0) / 7.0),
            "dow_cos": np.cos(2 * np.pi * (((timestamps.dayofweek + 1) % 7) + (timestamps.hour + timestamps.minute / 60.0 + timestamps.second / 3600.0) / 24.0) / 7.0),
            "lag_1": np.nan,
            "lag_5": np.nan,
            "lag_15": np.nan,
            "lag_60": np.nan,
            "lag_1440": np.nan,
            "rolling_mean_5": np.nan,
            "rolling_std_5": np.nan,
            "rolling_max_5": np.nan,
            "rolling_min_5": np.nan,
            "rolling_mean_15": np.nan,
            "rolling_std_15": np.nan,
            "rolling_max_15": np.nan,
            "rolling_min_15": np.nan,
            "rolling_mean_60": np.nan,
            "rolling_std_60": np.nan,
            "rolling_max_60": np.nan,
            "rolling_min_60": np.nan,
            "rolling_mean_240": np.nan,
            "rolling_std_240": np.nan,
            "rolling_max_240": np.nan,
            "rolling_min_240": np.nan,
            "rolling_mean_1440": np.nan,
            "rolling_std_1440": np.nan,
            "rolling_max_1440": np.nan,
            "rolling_min_1440": np.nan,
            "delta_5": np.nan,
            "delta_15": np.nan,
            "delta_60": np.nan,
            "delta_1440": np.nan,
            "slope_5": np.nan,
            "slope_15": np.nan,
            "slope_60": np.nan,
        }
    )
    suffix = {"10s": "10s", "30s": "30s", "1min": "1m", "5min": "5m", "10min": "10m", "15min": "15m"}[resolution]
    df.to_parquet(data_dir / f"power_load_{suffix}_all_features.parquet", index=False)


def _write_config_dir(tmp_path: Path) -> Path:
    """Materialize a self-contained config directory whose paths point at the test sandbox."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    pipeline_text = Path("config/pipeline.toml").read_text(encoding="utf-8")
    pipeline_text = pipeline_text.replace('gold_dir = "data/003_gold"', f'gold_dir = "{(tmp_path / "data" / "003_gold").as_posix()}"')
    pipeline_text = pipeline_text.replace('outputs_modeling_dir = "outputs/004_modeling"', f'outputs_modeling_dir = "{(tmp_path / "outputs" / "004_modeling").as_posix()}"')
    pipeline_text = pipeline_text.replace('outputs_performance_dir = "outputs/005_performance"', f'outputs_performance_dir = "{(tmp_path / "outputs" / "005_performance").as_posix()}"')
    (config_dir / "pipeline.toml").write_text(pipeline_text, encoding="utf-8")

    multires_text = Path("config/multires.toml").read_text(encoding="utf-8")
    multires_text = multires_text.replace('outputs_multires_dir = "outputs/006_multires"', f'outputs_multires_dir = "{(tmp_path / "outputs" / "006_multires").as_posix()}"')
    multires_text = multires_text.replace('outputs_rollout_dir = "outputs/007_rollout"', f'outputs_rollout_dir = "{(tmp_path / "outputs" / "007_rollout").as_posix()}"')
    multires_text = multires_text.replace('outputs_horizon_curve_dir = "outputs/009_horizon_curve"', f'outputs_horizon_curve_dir = "{(tmp_path / "outputs" / "009_horizon_curve").as_posix()}"')
    (config_dir / "multires.toml").write_text(multires_text, encoding="utf-8")

    (config_dir / "eda.toml").write_text(Path("config/eda.toml").read_text(encoding="utf-8"), encoding="utf-8")
    (config_dir / "modeling.toml").write_text(
        Path("config/modeling.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return config_dir


def _write_stage5_holdout_artifact(
    tmp_path: Path,
    *,
    learned_label: str = "1min/curated_ramp/hgb-balanced/residual+blend",
    learned_mae: float = 175.432301,
    persistence_mae: float = 173.724099,
) -> None:
    """Write the smallest Stage-5 holdout artifact set needed by downstream horizon-curve tests."""
    holdout_dir = _scoped_latest_dir(tmp_path, "005_performance")
    holdout_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "candidate_label": learned_label,
                "candidate_type": "learned",
                "resolution": "1min",
                "feature_set": "curated_ramp",
                "model_label": "hgb-balanced",
                "mae": learned_mae,
                "rmse": 250.0,
                "n_eval": 1440,
            },
            {
                "candidate_label": "persistence",
                "candidate_type": "baseline",
                "resolution": "1min",
                "feature_set": "baseline",
                "model_label": "persistence",
                "mae": persistence_mae,
                "rmse": 248.0,
                "n_eval": 1440,
            },
            {
                "candidate_label": "avg_workday",
                "candidate_type": "baseline",
                "resolution": "1min",
                "feature_set": "baseline",
                "model_label": "avg_workday",
                "mae": 180.0,
                "rmse": 255.0,
                "n_eval": 1440,
            },
        ]
    ).to_csv(holdout_dir / "holdout_evaluation.csv", index=False)


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
    """Seed a saved Stage-7 rollout run so registry and fallback behavior can be tested."""
    output_root = _normalize_rollout_output_root(output_root)
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
    pd.DataFrame(
        [
            {
                "selection_target": "endpoint_mae",
                "winner_candidate_label": (
                    model_label
                    if learned_endpoint_mae <= min(persistence_endpoint_mae, avg_workday_endpoint_mae)
                    else ("persistence" if persistence_endpoint_mae <= avg_workday_endpoint_mae else "avg_workday")
                ),
                "winner_metric_value": min(learned_endpoint_mae, persistence_endpoint_mae, avg_workday_endpoint_mae),
                "supporting_endpoint_mae": min(learned_endpoint_mae, persistence_endpoint_mae, avg_workday_endpoint_mae),
                "supporting_path_mae": min(learned_path_mae, persistence_path_mae, avg_workday_path_mae),
                "origin_n": origins_per_run,
                "decision_reason": "Lowest endpoint MAE across rollout candidates.",
            },
            {
                "selection_target": "path_mae",
                "winner_candidate_label": (
                    model_label
                    if learned_path_mae <= min(persistence_path_mae, avg_workday_path_mae)
                    else ("persistence" if persistence_path_mae <= avg_workday_path_mae else "avg_workday")
                ),
                "winner_metric_value": min(learned_path_mae, persistence_path_mae, avg_workday_path_mae),
                "supporting_endpoint_mae": min(learned_endpoint_mae, persistence_endpoint_mae, avg_workday_endpoint_mae),
                "supporting_path_mae": min(learned_path_mae, persistence_path_mae, avg_workday_path_mae),
                "origin_n": origins_per_run,
                "decision_reason": "Lowest path MAE across rollout candidates.",
            },
        ]
    ).to_csv(run_dir / "rollout_selection_summary.csv", index=False)


def _write_rollout_challenger_sweep(
    output_root: Path,
    *,
    sweep_run_id: str,
    recommended_run_id: str,
    horizon_minutes: int,
    resolution: str,
    feature_set: str,
    model_label: str,
    candidate_label: str,
    selection_target: str,
    origin_policy: str,
    endpoint_mae: float,
    path_mae: float,
    persistence_endpoint_mae: float,
    persistence_path_mae: float,
    best_baseline_endpoint_label: str,
    best_baseline_endpoint_mae: float,
    best_baseline_path_label: str,
    best_baseline_path_mae: float,
    origins_per_run: int = 8,
    generated_at_utc: str = "2026-03-10T03:26:22.070646+00:00",
) -> None:
    """Write a minimal challenger-sweep artifact bundle for Stage-7/Stage-8 reuse tests."""
    output_root = _normalize_rollout_output_root(output_root)
    sweep_dir = output_root / "challenger_sweeps" / sweep_run_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    recommended_run_path = output_root / recommended_run_id
    pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "candidate_label": candidate_label,
                "resolution": resolution,
                "feature_set": feature_set,
                "model_label": model_label,
                "learned_target_mode": "raw",
                "requested_origin_policy": origin_policy,
                "run_id": recommended_run_id,
                "run_path": str(recommended_run_path).replace("\\", "/"),
                "selection_target": selection_target,
                "selection_metric_name": selection_target,
                "selection_metric_value": path_mae if selection_target == "path_mae" else endpoint_mae,
                "selection_metric_pct": 0.0,
                "endpoint_mae": endpoint_mae,
                "endpoint_mae_pct": 0.0,
                "path_mae": path_mae,
                "path_mae_pct": 0.0,
                "phase_mean_mae": path_mae,
                "phase_mean_mae_pct": 0.0,
                "mean_coverage": 1.0,
                "origin_n": origins_per_run,
                "persistence_endpoint_mae": persistence_endpoint_mae,
                "persistence_endpoint_mae_pct": 0.0,
                "persistence_path_mae": persistence_path_mae,
                "persistence_path_mae_pct": 0.0,
                "best_baseline_endpoint_label": best_baseline_endpoint_label,
                "best_baseline_endpoint_mae": best_baseline_endpoint_mae,
                "best_baseline_endpoint_mae_pct": 0.0,
                "best_baseline_path_label": best_baseline_path_label,
                "best_baseline_path_mae": best_baseline_path_mae,
                "best_baseline_path_mae_pct": 0.0,
                "beats_persistence_endpoint": endpoint_mae < persistence_endpoint_mae,
                "beats_persistence_path": path_mae < persistence_path_mae,
                "beats_best_baseline_endpoint": endpoint_mae < best_baseline_endpoint_mae,
                "beats_best_baseline_path": path_mae < best_baseline_path_mae,
                "secondary_metric_value": endpoint_mae if selection_target == "path_mae" else path_mae,
                "reason": "Seeded integration-test challenger sweep.",
            }
        ]
    ).to_csv(sweep_dir / "candidate_results.csv", index=False)
    (sweep_dir / "recommended_candidate.json").write_text(
        json.dumps(
            {
                "generated_at_utc": generated_at_utc,
                "load_type": "commercial_facility",
                "artifact_namespace": "commercial_facility",
                "selection_target": selection_target,
                "requested_horizon_minutes": horizon_minutes,
                "requested_origin_policies": [origin_policy],
                "recommended_origin_policy": origin_policy,
                "recommended_candidate_label": candidate_label,
                "recommended_resolution": resolution,
                "recommended_feature_set": feature_set,
                "recommended_model_label": model_label,
                "recommended_run_id": recommended_run_id,
                "recommended_run_path": str(recommended_run_path).replace("\\", "/"),
                "recommended_metric_value": path_mae if selection_target == "path_mae" else endpoint_mae,
                "recommended_metric_pct": 0.0,
            }
        ),
        encoding="utf-8",
    )


def _run_multires_smoke(config_dir: Path) -> subprocess.CompletedProcess[str]:
    """Execute the Stage-6 smoke CLI against the sandbox config directory."""
    env = os.environ.copy()
    env["ELF_CONFIG_DIR"] = str(config_dir)
    return subprocess.run(
        [
            sys.executable,
            "scripts/005_multires_compare.py",
            "--mode",
            "smoke",
            "--resolution",
            "1min",
            "--horizon",
            "15",
            "--feature-set",
            "minimal",
            "--model-label",
            "ridge-medium",
            "--n-folds",
            "1",
            "--val-window-days",
            "1",
            "--origins-per-fold",
            "2",
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_rollout(config_dir: Path) -> subprocess.CompletedProcess[str]:
    """Execute a direct Stage-7 rollout run against the sandbox config directory."""
    env = os.environ.copy()
    env["ELF_CONFIG_DIR"] = str(config_dir)
    return subprocess.run(
        [
            sys.executable,
            "scripts/006_recursive_rollout.py",
            "--resolution",
            "5min",
            "--feature-set",
            "minimal",
            "--model-label",
            "ridge-medium",
            "--horizon-minutes",
            "60",
            "--origins",
            "1",
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_rollout_autoselect(
    config_dir: Path,
    *,
    horizon_minutes: int,
    origins: int = 1,
    selection_target: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute Stage-7 auto-selection for the requested horizon and optional objective."""
    env = os.environ.copy()
    env["ELF_CONFIG_DIR"] = str(config_dir)
    command = [
        sys.executable,
        "scripts/006_recursive_rollout.py",
        "--horizon-minutes",
        str(horizon_minutes),
        "--origins",
        str(origins),
    ]
    if selection_target is not None:
        command.extend(["--selection-target", selection_target])
    return subprocess.run(
        command,
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_rollout_sweep(
    config_dir: Path,
    *,
    horizon_minutes: int,
    origins: int = 1,
    selection_target: str = "path_mae",
    max_candidates: int = 3,
) -> subprocess.CompletedProcess[str]:
    """Execute the challenger-sweep CLI with a small candidate budget for integration tests."""
    env = os.environ.copy()
    env["ELF_CONFIG_DIR"] = str(config_dir)
    return subprocess.run(
        [
            sys.executable,
            "scripts/007_rollout_challenger_sweep.py",
            "--horizon-minutes",
            str(horizon_minutes),
            "--origins",
            str(origins),
            "--selection-target",
            selection_target,
            "--max-candidates",
            str(max_candidates),
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_horizon_curve(
    config_dir: Path,
    *,
    horizons: list[int],
    origins: int = 1,
    selection_target: str = "path_mae",
    max_candidates: int = 2,
) -> subprocess.CompletedProcess[str]:
    """Execute the horizon-curve CLI for the requested horizons inside the sandbox."""
    env = os.environ.copy()
    env["ELF_CONFIG_DIR"] = str(config_dir)
    command = [
        sys.executable,
        "scripts/008_horizon_curve.py",
        "--origins",
        str(origins),
        "--selection-target",
        selection_target,
        "--max-candidates",
        str(max_candidates),
    ]
    for horizon in horizons:
        command.extend(["--horizon-minutes", str(horizon)])
    return subprocess.run(
        command,
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_multires_compare_cli_writes_required_artifacts(tmp_path):
    """Stage-6 smoke runs should emit the core artifact bundle plus the latest alias."""
    config_dir = _write_config_dir(tmp_path)
    _write_gold(config_dir, "1min", periods_per_day=24 * 60, freq="1min")
    result = _run_multires_smoke(config_dir)

    assert result.returncode == 0, result.stderr
    latest = _scoped_latest_dir(tmp_path, "006_multires")
    assert (latest / "run_manifest.json").exists()
    assert (latest / "matched_horizon_metrics.csv").exists()
    assert (latest / "selection_summary.csv").exists()
    assert (latest / "winner_registry.csv").exists()


def test_multires_compare_cli_emits_contract_schemas(tmp_path):
    """Stage-6 smoke outputs should honor the documented CSV and manifest contracts."""
    config_dir = _write_config_dir(tmp_path)
    _write_gold(config_dir, "1min", periods_per_day=24 * 60, freq="1min")
    result = _run_multires_smoke(config_dir)

    assert result.returncode == 0, result.stderr
    latest = _scoped_latest_dir(tmp_path, "006_multires")
    manifest = json.loads((latest / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "006_multires"
    assert manifest["comparison_mode"] == "matched_horizon"
    assert manifest["source_mode"] == "bronze_direct"
    assert manifest["parallel_runtime"]["config"]["backend"] in {"threading", "loky", "sequential"}
    assert manifest["parallel_runtime"]["resolved_plans"]
    assert not any(
        warning.startswith("deduplicated_")
        for warning in manifest["warnings"]
    )
    assert "matched_horizon_metrics" in manifest["artifacts"]
    assert "selection_summary_csv" in manifest["artifacts"]
    assert "winner_registry" in manifest["artifacts"]

    matched = pd.read_csv(latest / "matched_horizon_metrics.csv")
    assert {
        "resolution",
        "horizon_minutes",
        "feature_set",
        "model_label",
        "forecast_strategy",
        "baseline_label",
        "mae",
        "mae_pct",
        "rmse",
        "path_mae_pct",
        "mae_ratio_to_persistence",
        "rmse_ratio_to_persistence",
        "n_eval",
        "eval_coverage",
        "runtime_seconds",
        "eligible",
        "source_mode",
    }.issubset(set(matched.columns))

    health = pd.read_csv(latest / "resolution_health.csv")
    assert {
        "resolution",
        "horizon_minutes",
        "feature_set",
        "model_label",
        "source_mode",
        "n_train",
        "n_validate",
        "n_test",
        "n_eval",
        "eval_coverage",
        "warmup_loss_pct",
        "runtime_seconds",
        "status",
        "failure_reason",
    }.issubset(set(health.columns))

    summary = pd.read_csv(latest / "selection_summary.csv")
    assert {
        "use_case",
        "winner_type",
        "winner_resolution",
        "winner_feature_set",
        "winner_model_label",
        "winner_forecast_strategy",
        "winner_horizon_minutes",
        "winner_endpoint_mae",
        "winner_endpoint_mae_pct",
        "winner_path_mae",
        "winner_path_mae_pct",
        "decision_reason",
        "practical_gain_passed",
        "pareto_passed",
    }.issubset(set(summary.columns))
    learned_strategies = set(
        matched.loc[matched["candidate_type"] == "learned", "forecast_strategy"].dropna().tolist()
    )
    assert {"recursive", "direct_endpoint"}.issubset(learned_strategies)

    fold_metrics = pd.read_csv(latest / "fold_metrics.csv")
    origin_metrics = pd.read_csv(latest / "origin_metrics.csv")
    assert int(fold_metrics.duplicated().sum()) == 0
    assert int(origin_metrics.duplicated().sum()) == 0

    winner_registry = pd.read_csv(latest / "winner_registry.csv")
    assert {
        "run_id",
        "winner_resolution",
        "winner_model_label",
        "winner_forecast_strategy",
        "winner_horizon_minutes",
        "winner_endpoint_mae",
        "winner_endpoint_mae_pct",
        "winner_path_mae",
        "winner_path_mae_pct",
    }.issubset(set(winner_registry.columns))


def test_multires_compare_skips_nonrepresentable_horizon_pairs_with_manifest_warning(tmp_path):
    """Stage-6 should skip impossible resolution/horizon pairs and record the warning in the manifest."""
    config_dir = _write_config_dir(tmp_path)
    _write_gold(config_dir, "10min", periods_per_day=24 * 6, freq="10min")
    env = os.environ.copy()
    env["ELF_CONFIG_DIR"] = str(config_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/005_multires_compare.py",
            "--mode",
            "smoke",
            "--resolution",
            "10min",
            "--horizon",
            "15",
            "--feature-set",
            "minimal",
            "--model-label",
            "ridge-medium",
            "--n-folds",
            "1",
            "--val-window-days",
            "1",
            "--origins-per-fold",
            "1",
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    latest = _scoped_latest_dir(tmp_path, "006_multires")
    manifest = json.loads((latest / "run_manifest.json").read_text(encoding="utf-8"))
    assert "skipped_non_representable:10min:15" in manifest["warnings"]


def test_multires_compare_skips_missing_resolutions_with_manifest_warning(tmp_path):
    """Stage-6 should continue past missing configured resolutions when others remain runnable."""
    config_dir = _write_config_dir(tmp_path)
    _write_gold(config_dir, "1min", periods_per_day=24 * 60, freq="1min")
    env = os.environ.copy()
    env["ELF_CONFIG_DIR"] = str(config_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/005_multires_compare.py",
            "--mode",
            "smoke",
            "--resolution",
            "1s",
            "--resolution",
            "1min",
            "--horizon",
            "15",
            "--feature-set",
            "minimal",
            "--model-label",
            "ridge-medium",
            "--n-folds",
            "1",
            "--val-window-days",
            "1",
            "--origins-per-fold",
            "1",
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    latest = _scoped_latest_dir(tmp_path, "006_multires")
    manifest = json.loads((latest / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["resolutions"] == ["1min"]
    assert any(
        warning.startswith("skipped_missing_resolution:1s:")
        for warning in manifest["warnings"]
    )


def test_multires_compare_supports_second_level_matched_horizons(tmp_path):
    """Stage-6 matched-horizon logic should support second-based resolutions when representable."""
    config_dir = _write_config_dir(tmp_path)
    _write_gold(config_dir, "30s", periods_per_day=24 * 60 * 2, freq="30s")
    env = os.environ.copy()
    env["ELF_CONFIG_DIR"] = str(config_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/005_multires_compare.py",
            "--mode",
            "smoke",
            "--resolution",
            "30s",
            "--horizon",
            "15",
            "--feature-set",
            "minimal",
            "--model-label",
            "ridge-medium",
            "--n-folds",
            "1",
            "--val-window-days",
            "1",
            "--origins-per-fold",
            "1",
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    latest = _scoped_latest_dir(tmp_path, "006_multires")
    manifest = json.loads((latest / "run_manifest.json").read_text(encoding="utf-8"))
    assert "skipped_non_representable:30s:15" not in manifest["warnings"]
    matched = pd.read_csv(latest / "matched_horizon_metrics.csv")
    assert (matched["resolution"] == "30s").any()
    assert (matched["horizon_minutes"] == 15).any()


def test_recursive_rollout_cli_writes_required_artifacts(tmp_path):
    """Stage-7 direct rollout runs should emit the expected metrics, origins, and summary artifacts."""
    config_dir = _write_config_dir(tmp_path)
    _write_gold(config_dir, "5min", periods_per_day=24 * 12, freq="5min")
    result = _run_rollout(config_dir)

    assert result.returncode == 0, result.stderr
    latest = _scoped_latest_dir(tmp_path, "007_rollout")
    assert (latest / "run_manifest.json").exists()
    assert (latest / "recursive_rollout_metrics.csv").exists()
    assert (latest / "recursive_rollout_by_origin.csv").exists()
    assert (latest / "selected_origins.csv").exists()
    assert (latest / "rollout_selection_summary.csv").exists()


def test_recursive_rollout_cli_emits_contract_schemas(tmp_path):
    """Stage-7 rollout outputs should match the documented manifest and CSV schemas."""
    config_dir = _write_config_dir(tmp_path)
    _write_gold(config_dir, "5min", periods_per_day=24 * 12, freq="5min")
    result = _run_rollout(config_dir)

    assert result.returncode == 0, result.stderr
    latest = _scoped_latest_dir(tmp_path, "007_rollout")
    manifest = json.loads((latest / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "007_rollout"
    assert manifest["strategy"] == "recursive"
    assert manifest["origin_policy"] in {"uniform", "midnight", "phase_balanced"}
    assert manifest["baseline_labels"] == [
        "persistence",
        "previous_day",
        "avg_workday",
        "anchored_workday",
        "hybrid_workday",
    ]
    assert "recursive_rollout_metrics" in manifest["artifacts"]
    assert "rollout_selection_summary_csv" in manifest["artifacts"]
    assert "rollout_registry" in manifest["artifacts"]
    assert "selection_context" in manifest["artifacts"]
    assert "selected_origins" in manifest["artifacts"]
    assert (latest / "rollout_registry.csv").exists()

    metrics = pd.read_csv(latest / "recursive_rollout_metrics.csv")
    assert {
        "candidate_label",
        "endpoint_mae",
        "endpoint_mae_pct",
        "endpoint_rmse",
        "path_mae",
        "path_mae_pct",
        "mean_coverage",
        "origin_n",
    }.issubset(set(metrics.columns))

    by_origin = pd.read_csv(latest / "recursive_rollout_by_origin.csv")
    assert {
        "origin_timestamp",
        "candidate_label",
        "endpoint_abs_error",
        "endpoint_sq_error",
        "endpoint_actual_abs",
        "path_mae",
        "path_rmse",
        "path_abs_error_sum",
        "path_actual_abs_sum",
        "coverage",
        "n_eval",
    }.issubset(set(by_origin.columns))

    health = pd.read_csv(latest / "rollout_health.csv")
    assert {
        "resolution",
        "feature_set",
        "model_label",
        "horizon_minutes",
        "origin_count",
        "origin_policy",
        "runtime_seconds",
        "status",
        "failure_reason",
    }.issubset(set(health.columns))

    summary = pd.read_csv(latest / "rollout_selection_summary.csv")
    assert {
        "selection_target",
        "winner_candidate_label",
        "winner_metric_value",
        "winner_metric_pct",
        "supporting_endpoint_mae",
        "supporting_endpoint_mae_pct",
        "supporting_path_mae",
        "supporting_path_mae_pct",
        "origin_n",
        "decision_reason",
    }.issubset(set(summary.columns))


def test_recursive_rollout_cli_falls_back_to_config_when_stage6_winner_horizon_does_not_match(tmp_path):
    """Stage-7 should use config defaults when Stage-6 has no exact winner for the requested horizon."""
    config_dir = _write_config_dir(tmp_path)
    _write_gold(config_dir, "5min", periods_per_day=24 * 12, freq="5min")
    _write_gold(config_dir, "10min", periods_per_day=24 * 6, freq="10min")
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
                "winner_horizon_minutes": 30,
                "decision_reason": "selected",
                "practical_gain_passed": True,
                "pareto_passed": True,
            }
        ]
    ).to_csv(latest_dir / "selection_summary.csv", index=False)

    result = _run_rollout_autoselect(config_dir, horizon_minutes=1440, origins=1)

    assert result.returncode == 0, result.stderr
    selection = json.loads((_scoped_latest_dir(tmp_path, "007_rollout") / "selection_context.json").read_text(encoding="utf-8"))
    assert selection["selection_source"] == "multires.toml"
    assert selection["selection_policy"] == "multires.toml"
    assert selection["requested_horizon_minutes"] == 1440
    assert selection["matched_stage6_horizon_minutes"] is None
    assert selection["matched_rollout_registry_horizon_minutes"] is None
    assert selection["resolution"] == "10min"
    assert selection["feature_set"] == "minimal"
    assert selection["model_label"] == "hgb-balanced"


def test_recursive_rollout_cli_uses_winner_registry_when_latest_is_narrower(tmp_path):
    """Stage-7 should reuse the cross-run Stage-6 winner registry instead of trusting latest blindly."""
    config_dir = _write_config_dir(tmp_path)
    _write_gold(config_dir, "5min", periods_per_day=24 * 12, freq="5min")
    latest_dir = tmp_path / "outputs" / "006_multires" / "latest"
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
                "winner_feature_set": "minimal",
                "winner_model_label": "ridge-medium",
                "winner_forecast_strategy": "recursive",
                "winner_horizon_minutes": 120,
                "decision_reason": "selected",
                "practical_gain_passed": True,
                "pareto_passed": True,
            }
        ]
    ).to_csv(tmp_path / "outputs" / "006_multires" / "winner_registry.csv", index=False)

    result = _run_rollout_autoselect(config_dir, horizon_minutes=120, origins=1)

    assert result.returncode == 0, result.stderr
    selection = json.loads(
        (_scoped_latest_dir(tmp_path, "007_rollout") / "selection_context.json").read_text(encoding="utf-8")
    )
    assert selection["selection_source"].endswith("winner_registry.csv")
    assert selection["selection_policy"] == "stage6_exact_horizon"
    assert selection["selection_run_id"] == "20260307T133220706885Z"
    assert selection["selection_run_stage"] == "006_multires"
    assert selection["resolution"] == "5min"
    assert selection["feature_set"] == "minimal"
    assert selection["model_label"] == "ridge-medium"
    assert selection["matched_stage6_horizon_minutes"] == 120


def test_recursive_rollout_cli_uses_explicit_selection_run_id(tmp_path):
    """Stage-7 should honor an explicit Stage-6 run id and replay that exact saved winner."""
    config_dir = _write_config_dir(tmp_path)
    _write_gold(config_dir, "5min", periods_per_day=24 * 12, freq="5min")
    run_id = "20260308T010203000000Z"
    run_dir = tmp_path / "outputs" / "006_multires" / run_id
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

    env = os.environ.copy()
    env["ELF_CONFIG_DIR"] = str(config_dir)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/006_recursive_rollout.py",
            "--selection-run-id",
            run_id,
            "--horizon-minutes",
            "60",
            "--origins",
            "1",
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    selection = json.loads(
        (_scoped_latest_dir(tmp_path, "007_rollout") / "selection_context.json").read_text(encoding="utf-8")
    )
    assert selection["selection_run_id"] == run_id
    assert selection["selection_source"].endswith(f"{run_id}\\selection_summary.csv") or selection[
        "selection_source"
    ].endswith(f"{run_id}/selection_summary.csv")
    assert selection["selection_policy"] == "stage6_exact_horizon"
    assert selection["selection_run_stage"] == "006_multires"
    assert selection["resolution"] == "5min"
    assert selection["feature_set"] == "minimal"
    assert selection["model_label"] == "ridge-medium"


def test_recursive_rollout_cli_uses_rollout_registry_for_long_horizon_fallback(tmp_path):
    """Stage-7 should fall back to prior rollout-registry evidence for long-horizon reuse."""
    config_dir = _write_config_dir(tmp_path)
    _write_gold(config_dir, "5min", periods_per_day=24 * 12, freq="5min")
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

    result = _run_rollout_autoselect(config_dir, horizon_minutes=1440, origins=1, selection_target="path_mae")

    assert result.returncode == 0, result.stderr
    selection = json.loads(
        (_scoped_latest_dir(tmp_path, "007_rollout") / "selection_context.json").read_text(encoding="utf-8")
    )
    assert selection["selection_source"].endswith("rollout_registry.csv")
    assert selection["selection_policy"] == "stage7_rollout_registry"
    assert selection["selection_run_stage"] == "007_rollout"
    assert selection["selection_run_id"] == "20260309T194339879991Z"
    assert selection["matched_stage6_horizon_minutes"] is None
    assert selection["matched_rollout_registry_horizon_minutes"] == 1440
    assert selection["resolution"] == "5min"
    assert selection["feature_set"] == "minimal"
    assert selection["model_label"] == "hgb-frontier-lr010-leaf100"


def test_recursive_rollout_cli_executes_cross_candidate_portfolio_from_challenger_sweep_registry(tmp_path):
    """Stage-7 should materialize sweep-derived portfolio policies as runnable rollout selections."""
    config_dir = _write_config_dir(tmp_path)
    _write_gold(config_dir, "5min", periods_per_day=24 * 12, freq="5min")
    _write_gold(config_dir, "10min", periods_per_day=24 * 6, freq="10min")
    rollout_dir = tmp_path / "outputs" / "007_rollout"
    sweep_run_id = "20260311T011357499505Z"
    sweep_dir = rollout_dir / "challenger_sweeps" / sweep_run_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    recommended_candidate_path = sweep_dir / "recommended_candidate.json"
    recommended_candidate_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-03-11T01:17:28.401607+00:00",
                "load_type": "commercial_facility",
                "artifact_namespace": "commercial_facility",
                "selection_target": "next_lock_mae",
                "requested_horizon_minutes": 60,
                "requested_origin_policies": ["phase_balanced"],
                "recommended_origin_policy": "phase_balanced",
                "recommended_candidate_label": "cross_candidate_portfolio::phase_bucket_next_lock_policy",
                "recommended_resolution": "mixed",
                "recommended_feature_set": "portfolio",
                "recommended_model_label": "cross_candidate_portfolio",
                "recommended_target_mode": "phase_bucket_next_lock_policy",
                "recommended_source_type": "cross_candidate_phase_bucket_portfolio",
                "recommended_run_id": sweep_run_id,
                "recommended_run_path": str(sweep_dir).replace("\\", "/"),
                "recommended_metric_value": 253.104260,
                "recommended_metric_pct": 15.969845,
            }
        ),
        encoding="utf-8",
    )
    (sweep_dir / "portfolio_policy_candidates.json").write_text(
        json.dumps(
            [
                {
                    "candidate_label": "cross_candidate_portfolio::phase_bucket_next_lock_policy",
                    "target_mode": "phase_bucket_next_lock_policy",
                    "selection_target": "next_lock_mae",
                    "requested_origin_policy": "phase_balanced",
                    "phase_bucket_mapping": {
                        "0": {
                            "run_id": "source-5min",
                            "candidate_label": "hgb-balanced::raw",
                            "resolution": "5min",
                            "feature_set": "minimal",
                            "model_label": "hgb-balanced",
                        },
                        "300": {
                            "run_id": "source-10min",
                            "candidate_label": "hgb-balanced::raw",
                            "resolution": "10min",
                            "feature_set": "minimal",
                            "model_label": "hgb-balanced",
                        },
                        "600": {
                            "run_id": "source-10min",
                            "candidate_label": "hgb-balanced::raw",
                            "resolution": "10min",
                            "feature_set": "minimal",
                            "model_label": "hgb-balanced",
                        },
                    },
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "sweep_run_id": sweep_run_id,
                "generated_at_utc": "2026-03-11T01:17:28.401607+00:00",
                "load_type": "commercial_facility",
                "artifact_namespace": "commercial_facility",
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
                "recommended_run_id": sweep_run_id,
                "recommended_run_path": str(sweep_dir).replace("\\", "/"),
                "recommended_metric_value": 253.104260,
                "recommended_metric_pct": 15.969845,
                "endpoint_mae": 751.696155,
                "endpoint_mae_pct": 35.234660,
                "path_mae": 496.893660,
                "path_mae_pct": 24.252664,
                "phase_mean_mae": 231.821351,
                "phase_mean_mae_pct": 10.554441,
                "next_lock_mae": 253.104260,
                "next_lock_mae_pct": 15.969845,
                "profile_shape_mae": 256.446567,
                "profile_shape_mae_pct": 12.562545,
                "energy_mae": 695.464054,
                "energy_mae_pct": 10.554441,
                "mean_coverage": 1.0,
                "origin_n": 8,
                "persistence_endpoint_mae": 444.904222,
                "persistence_endpoint_mae_pct": 19.714859,
                "persistence_path_mae": 389.439463,
                "persistence_path_mae_pct": 17.730532,
                "persistence_phase_mean_mae": 329.673630,
                "persistence_phase_mean_mae_pct": 15.009493,
                "persistence_next_lock_mae": 389.439463,
                "persistence_next_lock_mae_pct": 17.730532,
                "persistence_profile_shape_mae": 221.880580,
                "persistence_profile_shape_mae_pct": 10.101854,
                "best_baseline_endpoint_label": "persistence",
                "best_baseline_endpoint_mae": 444.904222,
                "best_baseline_endpoint_mae_pct": 19.714859,
                "best_baseline_path_label": "persistence",
                "best_baseline_path_mae": 389.439463,
                "best_baseline_path_mae_pct": 17.730532,
                "best_baseline_phase_label": "hybrid_workday",
                "best_baseline_phase_mae": 315.946407,
                "best_baseline_phase_mae_pct": 14.384515,
                "best_baseline_next_lock_label": "persistence",
                "best_baseline_next_lock_mae": 389.439463,
                "best_baseline_next_lock_mae_pct": 17.730532,
                "best_baseline_profile_shape_label": "persistence",
                "best_baseline_profile_shape_mae": 221.880580,
                "best_baseline_profile_shape_mae_pct": 10.101854,
                "beats_persistence_endpoint": False,
                "beats_persistence_path": False,
                "beats_persistence_phase": True,
                "beats_persistence_next_lock": True,
                "beats_persistence_profile_shape": False,
                "beats_best_baseline_endpoint": False,
                "beats_best_baseline_path": False,
                "beats_best_baseline_phase": True,
                "beats_best_baseline_next_lock": True,
                "beats_best_baseline_profile_shape": False,
                "candidate_results_path": str(sweep_dir / "candidate_results.csv").replace("\\", "/"),
                "recommended_candidate_path": str(recommended_candidate_path).replace("\\", "/"),
                "sweep_path": str(sweep_dir).replace("\\", "/"),
            }
        ]
    ).to_csv(rollout_dir / "challenger_sweep_registry.csv", index=False)

    result = _run_rollout_autoselect(
        config_dir,
        horizon_minutes=60,
        origins=2,
        selection_target="next_lock_mae",
    )

    assert result.returncode == 0, result.stderr
    latest_dir = _scoped_latest_dir(tmp_path, "007_rollout")
    selection = json.loads((latest_dir / "selection_context.json").read_text(encoding="utf-8"))
    assert selection["selection_policy"] == "stage7_challenger_sweep_registry"
    assert selection["selection_run_stage"] == "007_rollout_challenger_sweep"
    assert selection["resolution"] == "mixed"
    assert selection["feature_set"] == "portfolio"
    assert selection["model_label"] == "cross_candidate_portfolio"
    manifest = json.loads((latest_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["origin_selection_scope"] == "shared_timestamp_intersection"
    assert manifest["resolution"] == "mixed"
    assert manifest["selection_run_stage"] == "007_rollout_challenger_sweep"
    assert (latest_dir / "portfolio_policy_candidate.json").exists()
    metrics = pd.read_csv(latest_dir / "recursive_rollout_metrics.csv")
    assert metrics["candidate_label"].astype("string").eq(
        "cross_candidate_portfolio::phase_bucket_next_lock_policy"
    ).any()
    root_registry = pd.read_csv(_scoped_output_dir(tmp_path, "007_rollout") / "rollout_registry.csv")
    assert root_registry["model_label"].astype("string").eq("cross_candidate_portfolio").any()


def test_multires_compare_supports_focus_60m_mode(tmp_path):
    """Stage-6 focus mode should restrict the run to 60-minute evaluation and refresh its alias."""
    config_dir = _write_config_dir(tmp_path)
    _write_gold(config_dir, "1min", periods_per_day=24 * 60, freq="1min")
    env = os.environ.copy()
    env["ELF_CONFIG_DIR"] = str(config_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/005_multires_compare.py",
            "--mode",
            "focus_60m",
            "--resolution",
            "1min",
            "--feature-set",
            "curated",
            "--model-label",
            "hgb-balanced",
            "--n-folds",
            "1",
            "--val-window-days",
            "1",
            "--origins-per-fold",
            "1",
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    latest = _scoped_latest_dir(tmp_path, "006_multires")
    latest_focus = _scoped_latest_dir(tmp_path, "006_multires", alias="latest_focus_60m")
    manifest = json.loads((latest / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "focus_60m"
    assert manifest["horizons_minutes"] == [60]
    assert (latest_focus / "selection_summary.csv").exists()


def test_rollout_challenger_sweep_cli_writes_summary_and_promotes_best_run(tmp_path):
    """Stage-7 sweeps should compare candidates on shared origins and promote the saved winner."""
    config_dir = _write_config_dir(tmp_path)
    _write_gold(config_dir, "5min", periods_per_day=24 * 12, freq="5min")
    _write_gold(config_dir, "10min", periods_per_day=24 * 6, freq="10min")
    multires_text = (config_dir / "multires.toml").read_text(encoding="utf-8")
    multires_text = multires_text.replace("include_stage6_registry = true", "include_stage6_registry = false")
    (config_dir / "multires.toml").write_text(multires_text, encoding="utf-8")

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

    result = _run_rollout_sweep(
        config_dir,
        horizon_minutes=1440,
        origins=1,
        selection_target="path_mae",
        max_candidates=2,
    )

    assert result.returncode == 0, result.stderr
    sweep_latest = _scoped_output_dir(tmp_path, "007_rollout") / "challenger_sweeps" / "latest"
    manifest = json.loads((sweep_latest / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "007_rollout_challenger_sweep"
    assert manifest["candidate_count"] >= 1
    assert manifest["origin_selection_scope"] == "shared_timestamp_intersection"
    assert (sweep_latest / "shared_origins.csv").exists()
    candidate_results = pd.read_csv(sweep_latest / "candidate_results.csv")
    assert {
        "candidate_label",
        "selection_metric_value",
        "selection_metric_pct",
        "endpoint_mae_pct",
        "path_mae_pct",
        "run_id",
    }.issubset(set(candidate_results.columns))
    recommended = json.loads((sweep_latest / "recommended_candidate.json").read_text(encoding="utf-8"))
    assert recommended["recommended_metric_pct"] >= 0.0
    candidate_origin_sets = []
    for run_id in candidate_results["run_id"].astype("string").dropna().unique().tolist():
        selected_origins = pd.read_csv(_scoped_output_dir(tmp_path, "007_rollout") / str(run_id) / "selected_origins.csv")
        candidate_origin_sets.append(tuple(selected_origins["origin_timestamp"].astype("string").tolist()))
    assert len(set(candidate_origin_sets)) == 1
    rollout_latest = _scoped_latest_dir(tmp_path, "007_rollout")
    latest_manifest = json.loads((rollout_latest / "run_manifest.json").read_text(encoding="utf-8"))
    assert latest_manifest["stage"] == "007_rollout"
    assert (rollout_latest / "rollout_registry.csv").exists()


def test_horizon_curve_cli_writes_curve_artifacts_for_stage5_and_rollout_horizons(tmp_path):
    """Stage-8 should combine Stage-5 anchors with rollout evidence into the published curve artifacts."""
    config_dir = _write_config_dir(tmp_path)
    _write_gold(config_dir, "10min", periods_per_day=24 * 6, freq="10min")
    _write_stage5_holdout_artifact(tmp_path)
    _write_rollout_history_run(
        tmp_path / "outputs" / "007_rollout",
        run_id="20260309T194339879991Z",
        resolution="10min",
        feature_set="minimal",
        model_label="hgb-balanced",
        horizon_minutes=1440,
        origin_policy="uniform",
        selection_target="path_mae",
        learned_endpoint_mae=968.909580,
        learned_path_mae=783.077104,
        persistence_endpoint_mae=1119.137272,
        persistence_path_mae=1010.620668,
        avg_workday_endpoint_mae=986.676302,
        avg_workday_path_mae=850.145715,
        origins_per_run=8,
    )
    _write_rollout_challenger_sweep(
        tmp_path / "outputs" / "007_rollout",
        sweep_run_id="20260310T011702560391Z",
        recommended_run_id="20260309T194339879991Z",
        horizon_minutes=1440,
        resolution="10min",
        feature_set="minimal",
        model_label="hgb-balanced",
        candidate_label="hgb-balanced::raw",
        selection_target="path_mae",
        origin_policy="uniform",
        endpoint_mae=968.909580,
        path_mae=783.077104,
        persistence_endpoint_mae=1119.137272,
        persistence_path_mae=1010.620668,
        best_baseline_endpoint_label="avg_workday",
        best_baseline_endpoint_mae=986.676302,
        best_baseline_path_label="avg_workday",
        best_baseline_path_mae=850.145715,
        origins_per_run=8,
    )

    result = _run_horizon_curve(config_dir, horizons=[1, 1440], origins=1, max_candidates=2)

    assert result.returncode == 0, result.stderr
    latest = _scoped_latest_dir(tmp_path, "009_horizon_curve")
    assert (latest / "run_manifest.json").exists()
    assert (latest / "horizon_curve_summary.csv").exists()
    assert (latest / "horizon_curve_candidates.csv").exists()
    assert (latest / "horizon_curve_summary.md").exists()
    assert (latest / "crossover_summary.json").exists()
    assert (latest / "fig_horizon_ratio_curve.png").exists()
    assert (latest / "fig_horizon_absolute_mae.png").exists()

    manifest = json.loads((latest / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "008_horizon_curve"
    assert manifest["selection_target"] == "path_mae"
    assert manifest["include_stage5_anchor"] is True

    summary = pd.read_csv(latest / "horizon_curve_summary.csv")
    assert summary["horizon_minutes"].tolist() == [1, 1440]
    one_minute = summary.loc[summary["horizon_minutes"].eq(1)].iloc[0]
    assert one_minute["source_stage"] == "005_performance"
    assert one_minute["selection_policy"] == "stage5_holdout_anchor"
    one_day = summary.loc[summary["horizon_minutes"].eq(1440)].iloc[0]
    assert one_day["source_stage"] == "007_rollout_challenger_sweep"
    assert one_day["resolution"] == "10min"
    assert one_day["feature_set"] == "minimal"
    assert isinstance(one_day["model_label"], str) and one_day["model_label"]
    assert isinstance(one_day["candidate_label"], str) and one_day["candidate_label"]
    assert str(one_day["selection_source"]).endswith("recommended_candidate.json")
    assert bool(one_day["beats_persistence_path"]) is True

    crossover = json.loads((latest / "crossover_summary.json").read_text(encoding="utf-8"))
    assert crossover["horizons_minutes"] == [1, 1440]
    assert 1440 in crossover["beats_persistence_path_horizons"]
