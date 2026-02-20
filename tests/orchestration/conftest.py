"""Pytest fixtures for orchestration test isolation."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_pipeline_log_file(monkeypatch, tmp_path):
    """Redirect pipeline file logs to per-test temp files.

    This prevents negative-path orchestration tests from polluting `logs/pipeline.log`
    in the repository workspace.
    """
    monkeypatch.setenv("ELF_PIPELINE_LOG_FILE", str(tmp_path / "pipeline.log"))
