"""E2E runner command-plan tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "run_pipeline.py").exists() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Unable to locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_run_e2e_module():
    path = SCRIPTS_DIR / "run_e2e.py"
    spec = importlib.util.spec_from_file_location("test_run_e2e_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_command_plan_full_includes_performance():
    module = _load_run_e2e_module()
    steps = module.build_command_plan(
        mode="full",
        include_performance=True,
        keep_notebook_output=False,
        pytest_args=[],
    )

    assert [step.name for step in steps] == ["pipeline", "notebooks", "pytest"]
    assert "--include-performance" in steps[0].command
    assert "full" in steps[0].command
    assert "--keep-output" not in steps[1].command
    assert steps[2].command[:3] == [module.sys.executable, "-m", "pytest"]


def test_build_command_plan_quick_without_performance():
    module = _load_run_e2e_module()
    steps = module.build_command_plan(
        mode="quick",
        include_performance=False,
        keep_notebook_output=True,
        pytest_args=["tests/unit/test_config.py"],
    )

    assert "--include-performance" not in steps[0].command
    assert "--keep-output" in steps[1].command
    assert "tests/unit/test_config.py" in steps[2].command
