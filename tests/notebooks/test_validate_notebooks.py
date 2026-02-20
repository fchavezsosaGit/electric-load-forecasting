"""Notebook validator script tests for success and CLI argument behavior."""

from __future__ import annotations

import argparse
import importlib.util
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
    module.validate_notebooks([notebook])

    assert calls == [notebook]


def test_validate_notebooks_main_failure(monkeypatch):
    """Ensure main returns non-zero when validation raises an exception."""
    module = _load_validate_notebooks_module()
    monkeypatch.setattr(module, "parse_args", lambda: argparse.Namespace(notebook=None))
    monkeypatch.setattr(
        module,
        "validate_notebooks",
        lambda _notebooks: (_ for _ in ()).throw(RuntimeError("notebook failure")),
    )

    assert module.main() == 1


def test_validate_notebooks_notebook_argument(monkeypatch, tmp_path):
    """Ensure --notebook arguments are resolved and passed to validator."""
    module = _load_validate_notebooks_module()
    notebook = tmp_path / "single.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    captured: list[list[Path]] = []
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(notebook=[str(notebook)]),
    )
    monkeypatch.setattr(
        module,
        "validate_notebooks",
        lambda notebooks: captured.append(notebooks),
    )

    assert module.main() == 0
    assert captured == [[notebook.resolve()]]


def test_validate_notebooks_silver_matrix_profiles(monkeypatch):
    """Ensure silver notebook validation runs the configured resolution/profile matrix."""
    module = _load_validate_notebooks_module()
    silver = Path("notebooks/002_silver_eda.ipynb")

    calls: list[tuple[Path, dict[str, str] | None]] = []

    def _capture(path: Path, env_overrides: dict[str, str] | None = None):
        calls.append((path, env_overrides))

    monkeypatch.setattr(module, "execute_notebook", _capture)
    module.validate_notebooks([silver])

    assert calls[0] == (silver, None)
    assert len(calls) == 1 + len(module.SILVER_VALIDATION_PROFILES)
    for idx, profile in enumerate(module.SILVER_VALIDATION_PROFILES, start=1):
        assert calls[idx] == (silver, profile)
