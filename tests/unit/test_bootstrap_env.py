"""Bootstrap environment helper tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "run_pipeline.py").exists() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Unable to locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_bootstrap_module():
    path = SCRIPTS_DIR / "bootstrap_env.py"
    spec = importlib.util.spec_from_file_location("test_bootstrap_env_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dependency_specifiers_include_dev_group():
    module = _load_bootstrap_module()
    base = module.dependency_specifiers(include_dev=False)
    dev = module.dependency_specifiers(include_dev=True)

    assert "numpy>=1.24,<3.0" in base
    assert "pytest>=7.0,<9.0" not in base
    assert "pytest>=7.0,<9.0" in dev
    assert len(dev) >= len(base)


def test_smoke_imports_include_notebook_tools_for_dev():
    module = _load_bootstrap_module()
    base = module.smoke_imports(include_dev=False)
    dev = module.smoke_imports(include_dev=True)

    assert "numpy" in base
    assert "pytest" not in base
    assert "pytest" in dev
    assert "nbconvert" in dev
    assert "nbformat" in dev
