"""Bootstrap environment helper tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Find the repository root so the bootstrap helper can be imported safely."""
    for candidate in [start, *start.parents]:
        if (candidate / "run_pipeline.py").exists() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Unable to locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_bootstrap_module():
    """Load the bootstrap helper module directly from disk for unit tests."""
    path = SCRIPTS_DIR / "bootstrap_env.py"
    spec = importlib.util.spec_from_file_location("test_bootstrap_env_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dependency_specifiers_include_dev_group():
    """Include dev dependencies only when the caller opts into them."""
    module = _load_bootstrap_module()
    base = module.dependency_specifiers(include_dev=False)
    dev = module.dependency_specifiers(include_dev=True)

    assert "numpy>=1.24,<3.0" in base
    assert "pytest>=7.0,<9.0" not in base
    assert "pytest>=7.0,<9.0" in dev
    assert len(dev) >= len(base)


def test_dependency_specifiers_include_acceleration_group_when_requested():
    """Expose accelerator dependencies only when the caller opts into them."""
    module = _load_bootstrap_module()
    base = module.dependency_specifiers_with_options(
        include_dev=False,
        include_acceleration=False,
    )
    accelerated = module.dependency_specifiers_with_options(
        include_dev=False,
        include_acceleration=True,
    )

    assert any(dep.startswith("xgboost>=") for dep in accelerated)
    assert not any(dep.startswith("xgboost>=") for dep in base)
    assert len(accelerated) >= len(base)


def test_smoke_imports_include_notebook_tools_for_dev():
    """Expose notebook and test imports only in the dev smoke-check surface."""
    module = _load_bootstrap_module()
    base = module.smoke_imports(include_dev=False)
    dev = module.smoke_imports(include_dev=True)

    assert "joblib" in base
    assert "numpy" in base
    assert "pytest" not in base
    assert "pytest" in dev
    assert "nbconvert" in dev
    assert "nbformat" in dev
    assert "threadpoolctl" in base


def test_smoke_imports_include_acceleration_tools_when_requested():
    """Expose optional accelerator smoke imports only in the acceleration surface."""
    module = _load_bootstrap_module()
    base = module.smoke_imports_with_options(
        include_dev=False,
        include_acceleration=False,
    )
    accelerated = module.smoke_imports_with_options(
        include_dev=False,
        include_acceleration=True,
    )

    assert "xgboost" not in base
    assert "xgboost" in accelerated
