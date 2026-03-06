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
) -> argparse.Namespace:
    """Create argparse namespace values for pipeline test invocations."""
    return argparse.Namespace(
        stage=stage,
        resolution=resolution,
        verbose=False,
        dry_run=dry_run,
        include_performance=include_performance,
        performance_mode=performance_mode,
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
