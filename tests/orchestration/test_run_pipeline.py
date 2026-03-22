"""Pipeline orchestrator tests for CLI behavior and stage execution flow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest


def _args(
    stage: str,
    dry_run: bool = False,
    resolution: str | None = None,
    include_performance: bool = False,
    performance_mode: str = "quick",
    include_multires: bool = False,
    multires_mode: str = "smoke",
    include_rollout: bool = False,
    include_rollout_sweep: bool = False,
    include_horizon_curve: bool = False,
    include_forecast_control: bool = False,
) -> argparse.Namespace:
    """Create argparse namespace values for pipeline test invocations."""
    return argparse.Namespace(
        stage=stage,
        resolution=resolution,
        verbose=False,
        dry_run=dry_run,
        include_performance=include_performance,
        performance_mode=performance_mode,
        include_multires=include_multires,
        multires_mode=multires_mode,
        include_rollout=include_rollout,
        include_rollout_sweep=include_rollout_sweep,
        include_horizon_curve=include_horizon_curve,
        include_forecast_control=include_forecast_control,
    )


def test_run_pipeline_dry_run_validates_without_running_stages(pipeline_module, tmp_path, monkeypatch):
    """Ensure dry-run performs validation without executing transform stages."""
    raw_path = tmp_path / "000_raw" / "P_data.mat"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.touch()

    monkeypatch.setitem(pipeline_module.PATHS, "raw_mat", raw_path)
    monkeypatch.setitem(pipeline_module.PATHS, "logs_dir", tmp_path / "logs")
    monkeypatch.setitem(pipeline_module.PATHS, "silver_dir", tmp_path / "silver")
    monkeypatch.setitem(pipeline_module.PATHS, "gold_dir", tmp_path / "gold")
    monkeypatch.setitem(pipeline_module.PATHS, "model_dir", tmp_path / "model")

    monkeypatch.setattr(pipeline_module, "parse_args", lambda: _args(stage="all", dry_run=True))
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("should not run")),
    )

    assert pipeline_module.main() == 0


def test_run_pipeline_stage_bronze_only(pipeline_module, monkeypatch):
    """Ensure bronze-only mode runs only the bronze stage."""
    calls: list[tuple[str, list[str] | None]] = []
    monkeypatch.setattr(pipeline_module, "parse_args", lambda: _args(stage="bronze"))
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_stage",
        lambda stage, resolutions: calls.append((stage, resolutions)) or [],
    )
    assert pipeline_module.main() == 0
    assert calls == [("bronze", None)]


def test_run_pipeline_stage_silver_only(pipeline_module, monkeypatch):
    """Ensure silver-only mode runs only the silver stage."""
    calls: list[tuple[str, list[str] | None]] = []
    monkeypatch.setattr(pipeline_module, "parse_args", lambda: _args(stage="silver"))
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_stage",
        lambda stage, resolutions: calls.append((stage, resolutions)) or [],
    )
    assert pipeline_module.main() == 0
    assert calls == [("silver", None)]


def test_run_pipeline_stage_gold_only(pipeline_module, monkeypatch):
    """Ensure gold-only mode runs only the gold stage."""
    calls: list[tuple[str, list[str] | None]] = []
    monkeypatch.setattr(pipeline_module, "parse_args", lambda: _args(stage="gold"))
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_stage",
        lambda stage, resolutions: calls.append((stage, resolutions)) or [],
    )
    assert pipeline_module.main() == 0


def test_run_pipeline_stage_modeling_only(pipeline_module, monkeypatch):
    """Ensure modeling-only mode dispatches to the modeling stage runner."""
    calls: list[str] = []
    monkeypatch.setattr(pipeline_module, "parse_args", lambda: _args(stage="modeling"))
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_modeling_stage",
        lambda: calls.append("modeling") or [],
    )
    assert pipeline_module.main() == 0
    assert calls == ["modeling"]


def test_run_pipeline_dry_run_performance_uses_project_scoped_manifest(
    pipeline_module, tmp_path, monkeypatch
):
    """Ensure performance dry-run resolves the modeling manifest against PROJECT_ROOT."""
    manifest_path = tmp_path / "outputs" / "004_modeling" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pipeline_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setitem(pipeline_module.PATHS, "logs_dir", tmp_path / "logs")
    monkeypatch.setitem(pipeline_module.PATHS, "silver_dir", tmp_path / "silver")
    monkeypatch.setitem(pipeline_module.PATHS, "gold_dir", tmp_path / "gold")
    monkeypatch.setitem(pipeline_module.PATHS, "model_dir", tmp_path / "model")

    pipeline_module._dry_run("performance", None)


def test_run_pipeline_stage_performance_only(pipeline_module, monkeypatch):
    """Ensure performance-only mode dispatches to performance stage runner."""
    calls: list[tuple[list[str] | None, str]] = []
    monkeypatch.setattr(pipeline_module, "parse_args", lambda: _args(stage="performance"))
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_performance_stage",
        lambda resolutions, performance_mode: calls.append((resolutions, performance_mode)) or [],
    )
    assert pipeline_module.main() == 0
    assert calls == [(None, "quick")]


def test_run_performance_stage_reports_latest_manifest_path(pipeline_module, monkeypatch):
    """Ensure Stage-5 returns the latest-alias manifest path rather than a flat root file."""
    monkeypatch.setattr(pipeline_module, "_ensure_step4_modeling_artifacts", lambda: [])
    monkeypatch.setattr(pipeline_module.subprocess, "run", lambda *args, **kwargs: None)

    outputs = pipeline_module._run_performance_stage(None, performance_mode="quick")

    assert outputs == [
        pipeline_module._project_scoped_path(
            pipeline_module.scoped_output_path(pipeline_module.PATHS["outputs_performance_dir"])
        )
        / "latest"
        / "run_manifest.json"
    ]


def test_run_pipeline_stage_multires_only(pipeline_module, monkeypatch):
    """Ensure multires-only mode dispatches to multires stage runner."""
    calls: list[tuple[list[str] | None, str]] = []
    monkeypatch.setattr(pipeline_module, "parse_args", lambda: _args(stage="multires"))
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_multires_stage",
        lambda resolutions, multires_mode: calls.append((resolutions, multires_mode)) or [],
    )
    assert pipeline_module.main() == 0
    assert calls == [(None, "smoke")]


def test_run_pipeline_stage_rollout_only(pipeline_module, monkeypatch):
    """Ensure rollout-only mode dispatches to rollout stage runner."""
    calls: list[list[str] | None] = []
    monkeypatch.setattr(pipeline_module, "parse_args", lambda: _args(stage="rollout"))
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_rollout_stage",
        lambda resolutions: calls.append(resolutions) or [],
    )
    assert pipeline_module.main() == 0
    assert calls == [None]


def test_run_pipeline_stage_rollout_sweep_only(pipeline_module, monkeypatch):
    """Ensure rollout-sweep-only mode dispatches to the challenger sweep runner."""
    calls: list[str] = []
    monkeypatch.setattr(pipeline_module, "parse_args", lambda: _args(stage="rollout_sweep"))
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_rollout_sweep_stage",
        lambda: calls.append("rollout_sweep") or [],
    )
    assert pipeline_module.main() == 0
    assert calls == ["rollout_sweep"]


def test_run_pipeline_stage_horizon_curve_only(pipeline_module, monkeypatch):
    """Ensure horizon-curve-only mode dispatches to the horizon-curve stage runner."""
    calls: list[str] = []
    monkeypatch.setattr(pipeline_module, "parse_args", lambda: _args(stage="horizon_curve"))
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_horizon_curve_stage",
        lambda: calls.append("horizon_curve") or [],
    )
    assert pipeline_module.main() == 0
    assert calls == ["horizon_curve"]


def test_run_pipeline_stage_all_can_include_performance(pipeline_module, monkeypatch):
    """Ensure --include-performance adds the performance stage after gold."""
    calls: list[tuple[str, list[str] | None]] = []
    perf_calls: list[tuple[list[str] | None, str]] = []
    monkeypatch.setattr(
        pipeline_module,
        "parse_args",
        lambda: _args(stage="all", include_performance=True, performance_mode="preflight"),
    )
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_stage",
        lambda stage, resolutions: calls.append((stage, resolutions)) or [],
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_performance_stage",
        lambda resolutions, performance_mode: perf_calls.append((resolutions, performance_mode)) or [],
    )
    assert pipeline_module.main() == 0
    assert calls == [("bronze", None), ("silver", None), ("gold", None)]
    assert perf_calls == [(None, "preflight")]


def test_run_pipeline_stage_all_can_include_multires_and_rollout(pipeline_module, monkeypatch):
    """Ensure multires and rollout are appended in correct order when requested."""
    calls: list[tuple[str, list[str] | None]] = []
    multires_calls: list[tuple[list[str] | None, str]] = []
    rollout_calls: list[list[str] | None] = []
    monkeypatch.setattr(
        pipeline_module,
        "parse_args",
        lambda: _args(
            stage="all",
            include_multires=True,
            multires_mode="candidate",
            include_rollout=True,
        ),
    )
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_stage",
        lambda stage, resolutions: calls.append((stage, resolutions)) or [],
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_multires_stage",
        lambda resolutions, multires_mode: multires_calls.append((resolutions, multires_mode)) or [],
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_rollout_stage",
        lambda resolutions: rollout_calls.append(resolutions) or [],
    )
    assert pipeline_module.main() == 0
    expected_resolutions = pipeline_module._merge_resolutions(
        list(pipeline_module.DEFAULT_RESOLUTIONS),
        list(pipeline_module.MULTIRES_PROFILES["candidate"]["resolutions"]),
        [pipeline_module.MULTIRES_ROLLOUT["selected_resolution"]],
    )
    assert calls == [("bronze", None), ("silver", expected_resolutions), ("gold", expected_resolutions)]
    assert multires_calls == [(None, "candidate")]
    assert rollout_calls == [None]


def test_run_pipeline_stage_all_can_include_rollout_sweep(pipeline_module, monkeypatch):
    """Ensure rollout sweep is appended after multires when requested."""
    calls: list[tuple[str, list[str] | None]] = []
    multires_calls: list[tuple[list[str] | None, str]] = []
    sweep_calls: list[str] = []
    monkeypatch.setattr(
        pipeline_module,
        "parse_args",
        lambda: _args(
            stage="all",
            include_multires=True,
            multires_mode="candidate",
            include_rollout_sweep=True,
        ),
    )
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_stage",
        lambda stage, resolutions: calls.append((stage, resolutions)) or [],
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_multires_stage",
        lambda resolutions, multires_mode: multires_calls.append((resolutions, multires_mode)) or [],
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_rollout_sweep_stage",
        lambda: sweep_calls.append("rollout_sweep") or [],
    )
    assert pipeline_module.main() == 0
    expected_resolutions = pipeline_module._merge_resolutions(
        list(pipeline_module.DEFAULT_RESOLUTIONS),
        list(pipeline_module.MULTIRES_PROFILES["candidate"]["resolutions"]),
    )
    assert calls == [("bronze", None), ("silver", expected_resolutions), ("gold", expected_resolutions)]
    assert multires_calls == [(None, "candidate")]
    assert sweep_calls == ["rollout_sweep"]


def test_run_pipeline_stage_all_can_include_horizon_curve(pipeline_module, monkeypatch):
    """Ensure horizon-curve runs after Stage-6 when explicitly requested."""
    calls: list[tuple[str, list[str] | None]] = []
    multires_calls: list[tuple[list[str] | None, str]] = []
    horizon_calls: list[str] = []
    monkeypatch.setattr(
        pipeline_module,
        "parse_args",
        lambda: _args(
            stage="all",
            include_horizon_curve=True,
            multires_mode="candidate",
        ),
    )
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_stage",
        lambda stage, resolutions: calls.append((stage, resolutions)) or [],
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_multires_stage",
        lambda resolutions, multires_mode: multires_calls.append((resolutions, multires_mode)) or [],
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_horizon_curve_stage",
        lambda: horizon_calls.append("horizon_curve") or [],
    )

    assert pipeline_module.main() == 0

    expected_resolutions = pipeline_module._merge_resolutions(
        list(pipeline_module.DEFAULT_RESOLUTIONS),
        list(pipeline_module.MULTIRES_PROFILES["candidate"]["resolutions"]),
        [pipeline_module.MULTIRES_ROLLOUT["selected_resolution"]],
    )
    assert calls == [("bronze", None), ("silver", expected_resolutions), ("gold", expected_resolutions)]
    assert multires_calls == [(None, "candidate")]
    assert horizon_calls == ["horizon_curve"]


def test_run_pipeline_stage_all_can_include_forecast_control(pipeline_module, monkeypatch):
    """Ensure forecast-control runs after the rollout stack when explicitly requested."""
    calls: list[tuple[str, list[str] | None]] = []
    multires_calls: list[tuple[list[str] | None, str]] = []
    forecast_calls: list[str] = []
    monkeypatch.setattr(
        pipeline_module,
        "parse_args",
        lambda: _args(
            stage="all",
            include_multires=True,
            multires_mode="candidate",
            include_forecast_control=True,
        ),
    )
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_stage",
        lambda stage, resolutions: calls.append((stage, resolutions)) or [],
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_multires_stage",
        lambda resolutions, multires_mode: multires_calls.append((resolutions, multires_mode)) or [],
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_forecast_control_stage",
        lambda: forecast_calls.append("forecast_control") or [],
    )

    assert pipeline_module.main() == 0

    expected_resolutions = pipeline_module._merge_resolutions(
        list(pipeline_module.DEFAULT_RESOLUTIONS),
        list(pipeline_module.MULTIRES_PROFILES["candidate"]["resolutions"]),
        [
            pipeline_module.MULTIRES_ROLLOUT["selected_resolution"],
            pipeline_module.MULTIRES_FORECAST_CONTROL["actual_resolution"],
        ],
    )
    assert calls == [("bronze", None), ("silver", expected_resolutions), ("gold", expected_resolutions)]
    assert multires_calls == [(None, "candidate")]
    assert forecast_calls == ["forecast_control"]


def test_run_pipeline_logs_pipeline_health_summary(pipeline_module, monkeypatch, capsys):
    """Ensure successful runs emit the standardized pipeline health summary."""
    monkeypatch.setattr(pipeline_module, "parse_args", lambda: _args(stage="bronze"))
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(pipeline_module, "_run_stage", lambda stage, resolutions: [])

    assert pipeline_module.main() == 0

    captured = capsys.readouterr()
    assert "PIPELINE HEALTH: PASS" in captured.err
    assert "completed=1/1" in captured.err


def test_ensure_step4_modeling_artifacts_reuses_existing_manifest(
    pipeline_module, tmp_path, monkeypatch
):
    """Ensure performance bootstrap is skipped when step-4 manifest already exists."""
    manifest_path = tmp_path / "outputs" / "004_modeling" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pipeline_module, "PROJECT_ROOT", tmp_path)
    called = {"value": False}
    monkeypatch.setattr(
        pipeline_module,
        "_run_modeling_stage",
        lambda: called.update(value=True) or [manifest_path],
    )

    outputs = pipeline_module._ensure_step4_modeling_artifacts()

    assert outputs == [manifest_path]
    assert called["value"] is False


def test_ensure_step4_modeling_artifacts_bootstraps_when_missing(
    pipeline_module, tmp_path, monkeypatch
):
    """Ensure performance bootstrap triggers modeling when step-4 manifest is absent."""
    monkeypatch.setattr(pipeline_module, "PROJECT_ROOT", tmp_path)
    manifest_path = tmp_path / "outputs" / "004_modeling" / "run_manifest.json"
    calls: list[str] = []

    def _run_modeling():
        calls.append("modeling")
        return [manifest_path]

    monkeypatch.setattr(pipeline_module, "_run_modeling_stage", _run_modeling)

    outputs = pipeline_module._ensure_step4_modeling_artifacts()

    assert outputs == [manifest_path]
    assert calls == ["modeling"]


def test_run_pipeline_resolution_limit_and_alias(pipeline_module, monkeypatch):
    """Ensure resolution aliases are normalized before stage execution."""
    calls: list[tuple[str, list[str] | None]] = []
    monkeypatch.setattr(
        pipeline_module,
        "parse_args",
        lambda: _args(stage="silver", resolution="60s"),
    )
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_stage",
        lambda stage, resolutions: calls.append((stage, resolutions)) or [],
    )
    assert pipeline_module.main() == 0
    assert calls == [("silver", ["1min"])]


def test_run_pipeline_invalid_resolution_returns_non_zero(pipeline_module, monkeypatch):
    """Ensure invalid resolution arguments return exit code 2."""
    monkeypatch.setattr(
        pipeline_module,
        "parse_args",
        lambda: _args(stage="silver", resolution="999min"),
    )
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    assert pipeline_module.main() == 2


def test_run_pipeline_failure_returns_non_zero(pipeline_module, monkeypatch):
    """Ensure stage failures return non-zero exit code."""
    monkeypatch.setattr(pipeline_module, "parse_args", lambda: _args(stage="bronze"))
    monkeypatch.setattr(pipeline_module, "validate_config", lambda: None)
    monkeypatch.setattr(
        pipeline_module,
        "_run_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert pipeline_module.main() == 1


def test_run_pipeline_logging_config_writes_pipeline_log(pipeline_module, tmp_path, monkeypatch):
    """Ensure logging configuration creates file and stream handlers."""
    monkeypatch.delenv("ELF_PIPELINE_LOG_FILE", raising=False)
    logs_dir = tmp_path / "logs"
    monkeypatch.setitem(pipeline_module.PATHS, "logs_dir", logs_dir)
    pipeline_module._configure_logging(verbose=False)
    logging_target = logs_dir / "pipeline.log"
    assert logging_target.exists()
    root_handlers = pipeline_module.logging.getLogger().handlers
    assert any(isinstance(handler, pipeline_module.logging.StreamHandler) for handler in root_handlers)
    assert any(isinstance(handler, pipeline_module.logging.FileHandler) for handler in root_handlers)


def test_run_pipeline_logging_config_respects_env_log_file_override(
    pipeline_module, tmp_path, monkeypatch
):
    """Ensure file logging honors ELF_PIPELINE_LOG_FILE path override."""
    override_log_path = tmp_path / "custom" / "pipeline_test.log"
    monkeypatch.setenv("ELF_PIPELINE_LOG_FILE", str(override_log_path))
    pipeline_module._configure_logging(verbose=False)

    assert override_log_path.exists()


def test_run_pipeline_logging_config_can_disable_file_handler(pipeline_module, monkeypatch):
    """Ensure file logging can be disabled via ELF_PIPELINE_LOG_FILE."""
    monkeypatch.setenv("ELF_PIPELINE_LOG_FILE", "off")
    pipeline_module._configure_logging(verbose=False)
    root_handlers = pipeline_module.logging.getLogger().handlers
    assert any(isinstance(handler, pipeline_module.logging.StreamHandler) for handler in root_handlers)
    assert not any(
        isinstance(handler, pipeline_module.logging.FileHandler) for handler in root_handlers
    )


def test_cli_invalid_stage_exits_with_error():
    """Ensure argparse rejects invalid stage options."""
    result = subprocess.run(
        [sys.executable, "run_pipeline.py", "--stage", "invalid_stage"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--stage" in result.stderr
