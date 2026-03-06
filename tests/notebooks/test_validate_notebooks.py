"""Notebook validator script tests for success and CLI argument behavior."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Resolve the repository root from a nested test location."""
    for candidate in [start, *start.parents]:
        if (candidate / "run_pipeline.py").exists() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Unable to locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_validate_notebooks_module():
    """Load the notebook validation script as an importable module."""
    path = SCRIPTS_DIR / "validate_notebooks.py"
    spec = importlib.util.spec_from_file_location("test_validate_notebooks_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_notebooks_success(monkeypatch, tmp_path):
    """Ensure validate_notebooks invokes execute_notebook per input notebook."""
    module = _load_validate_notebooks_module()
    notebook = tmp_path / "demo.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    calls: list[Path] = []
    monkeypatch.setattr(module, "execute_notebook", lambda path: calls.append(path))
    monkeypatch.setattr(module, "_clear_notebook_outputs", lambda path: None)
    module.validate_notebooks([notebook])

    assert calls == [notebook]


def test_validate_notebooks_main_failure(monkeypatch):
    """Ensure main returns non-zero when validation raises an exception."""
    module = _load_validate_notebooks_module()
    monkeypatch.setattr(module, "parse_args", lambda: argparse.Namespace(notebook=None, keep_output=False))
    monkeypatch.setattr(
        module,
        "validate_notebooks",
        lambda _notebooks, clear_outputs=True: (_ for _ in ()).throw(RuntimeError("notebook failure")),
    )

    assert module.main() == 1


def test_validate_notebooks_notebook_argument(monkeypatch, tmp_path):
    """Ensure --notebook arguments are resolved and passed to validator."""
    module = _load_validate_notebooks_module()
    notebook = tmp_path / "single.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    captured: list[tuple[list[Path], bool]] = []
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(notebook=[str(notebook)], keep_output=False),
    )
    monkeypatch.setattr(
        module,
        "validate_notebooks",
        lambda notebooks, clear_outputs=True: captured.append((notebooks, clear_outputs)),
    )

    assert module.main() == 0
    assert captured == [([notebook.resolve()], True)]


def test_validate_notebooks_silver_matrix_profiles(monkeypatch):
    """Ensure silver notebook validation runs the configured resolution/profile matrix."""
    module = _load_validate_notebooks_module()
    silver = Path("notebooks/002_silver_eda.ipynb")

    calls: list[tuple[Path, dict[str, str] | None]] = []

    def _capture(path: Path, env_overrides: dict[str, str] | None = None):
        calls.append((path, env_overrides))

    monkeypatch.setattr(module, "execute_notebook", _capture)
    monkeypatch.setattr(module, "_clear_notebook_outputs", lambda path: None)
    module.validate_notebooks([silver])

    assert calls[0] == (silver, None)
    assert len(calls) == 1 + len(module.SILVER_VALIDATION_PROFILES)
    for idx, profile in enumerate(module.SILVER_VALIDATION_PROFILES, start=1):
        assert calls[idx] == (silver, profile)


def test_validate_notebooks_default_scope_stops_at_modeling_notebook():
    """Ensure notebook smoke suite stays lean and ends at the integrated modeling notebook."""
    module = _load_validate_notebooks_module()
    names = [path.name for path in module.DEFAULT_NOTEBOOKS]
    assert names == [
        "000_raw_eda.ipynb",
        "001_bronze_eda.ipynb",
        "002_silver_eda.ipynb",
        "003_modeling.ipynb",
    ]


def test_validate_notebooks_keep_output_flag_disables_cleanup(monkeypatch, tmp_path):
    """Ensure CLI keep-output flag preserves notebook cell outputs."""
    module = _load_validate_notebooks_module()
    notebook = tmp_path / "single.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    captured: list[tuple[list[Path], bool]] = []
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(notebook=[str(notebook)], keep_output=True),
    )
    monkeypatch.setattr(
        module,
        "validate_notebooks",
        lambda notebooks, clear_outputs=True: captured.append((notebooks, clear_outputs)),
    )

    assert module.main() == 0
    assert captured == [([notebook.resolve()], False)]


def test_clear_notebook_outputs_removes_only_outputs(tmp_path):
    """Ensure output cleanup strips cell outputs while preserving execution counts."""
    module = _load_validate_notebooks_module()
    notebook = tmp_path / "demo.ipynb"
    payload = {
        "cells": [
            {
                "id": "demo-cell",
                "cell_type": "code",
                "execution_count": 7,
                "metadata": {},
                "outputs": [{"output_type": "stream", "name": "stdout", "text": "hello"}],
                "source": ["print('hello')"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook.write_text(json.dumps(payload), encoding="utf-8")

    module._clear_notebook_outputs(notebook)

    cleaned = json.loads(notebook.read_text(encoding="utf-8"))
    cell = cleaned["cells"][0]
    assert cell["outputs"] == []
    assert cell["execution_count"] == 7
