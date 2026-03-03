"""Pipeline orchestrator tests for CLI behavior and stage execution flow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest


def _args(stage: str, dry_run: bool = False, resolution: str | None = None) -> argparse.Namespace:
    """Create argparse namespace values for pipeline test invocations."""
    return argparse.Namespace(stage=stage, resolution=resolution, verbose=False, dry_run=dry_run)


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
    assert calls == [("gold", None)]


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
