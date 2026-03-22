"""E2E runner command-plan tests."""

from __future__ import annotations

from argparse import Namespace
import importlib.util
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Find the repository root so the E2E helper can be imported from tests."""
    for candidate in [start, *start.parents]:
        if (candidate / "run_pipeline.py").exists() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Unable to locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_run_e2e_module():
    """Load the E2E runner module directly from disk for command-plan tests."""
    path = SCRIPTS_DIR / "run_e2e.py"
    spec = importlib.util.spec_from_file_location("test_run_e2e_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_command_plan_full_includes_performance():
    """Include the performance stage when the full E2E mode requests it."""
    module = _load_run_e2e_module()
    steps = module.build_command_plan(
        mode="full",
        include_performance=True,
        include_multires=False,
        include_rollout=False,
        include_rollout_sweep=False,
        include_horizon_curve=False,
        include_forecast_control=False,
        keep_notebook_output=False,
        pytest_args=[],
    )

    assert [step.name for step in steps] == ["pipeline", "notebooks", "pytest"]
    assert "--include-performance" in steps[0].command
    assert "full" in steps[0].command
    assert "--keep-output" not in steps[1].command
    assert steps[2].command[:3] == [module.sys.executable, "-m", "pytest"]


def test_build_command_plan_quick_without_performance():
    """Allow quick-mode E2E plans to skip Stage-5 while forwarding other options."""
    module = _load_run_e2e_module()
    steps = module.build_command_plan(
        mode="quick",
        include_performance=False,
        include_multires=False,
        include_rollout=False,
        include_rollout_sweep=False,
        include_horizon_curve=False,
        include_forecast_control=False,
        keep_notebook_output=True,
        pytest_args=["tests/unit/test_config.py"],
    )

    assert "--include-performance" not in steps[0].command
    assert "--keep-output" in steps[1].command
    assert "tests/unit/test_config.py" in steps[2].command


def test_build_command_plan_can_include_multires_and_rollout():
    """Add multiresolution and rollout stages when explicitly requested."""
    module = _load_run_e2e_module()
    steps = module.build_command_plan(
        mode="full",
        include_performance=True,
        include_multires=True,
        include_rollout=True,
        include_rollout_sweep=False,
        include_horizon_curve=False,
        include_forecast_control=False,
        keep_notebook_output=False,
        pytest_args=[],
    )

    assert "--include-multires" in steps[0].command
    assert "--include-rollout" in steps[0].command
    assert "candidate" in steps[0].command


def test_build_command_plan_rollout_only_does_not_force_multires():
    """Keep rollout opt-in independent from Stage-6 multires execution."""
    module = _load_run_e2e_module()
    steps = module.build_command_plan(
        mode="quick",
        include_performance=True,
        include_multires=False,
        include_rollout=True,
        include_rollout_sweep=False,
        include_horizon_curve=False,
        include_forecast_control=False,
        keep_notebook_output=False,
        pytest_args=[],
    )

    assert "--include-rollout" in steps[0].command
    assert "--include-multires" not in steps[0].command


def test_build_command_plan_can_include_horizon_curve():
    """Add the Stage-8 horizon-curve flag when requested."""
    module = _load_run_e2e_module()
    steps = module.build_command_plan(
        mode="quick",
        include_performance=True,
        include_multires=False,
        include_rollout=False,
        include_rollout_sweep=False,
        include_horizon_curve=True,
        include_forecast_control=False,
        keep_notebook_output=False,
        pytest_args=[],
    )

    assert "--include-horizon-curve" in steps[0].command


def test_build_command_plan_can_include_forecast_control():
    """Add the Stage-10 forecast-control flag when requested."""
    module = _load_run_e2e_module()
    steps = module.build_command_plan(
        mode="quick",
        include_performance=True,
        include_multires=False,
        include_rollout=False,
        include_rollout_sweep=False,
        include_horizon_curve=False,
        include_forecast_control=True,
        keep_notebook_output=False,
        pytest_args=[],
    )

    assert "--include-forecast-control" in steps[0].command


def test_build_command_plan_can_include_rollout_sweep():
    """Add the standalone Stage-7 challenger sweep when the operator requests it."""
    module = _load_run_e2e_module()
    steps = module.build_command_plan(
        mode="quick",
        include_performance=True,
        include_multires=False,
        include_rollout=False,
        include_rollout_sweep=True,
        include_horizon_curve=False,
        include_forecast_control=False,
        keep_notebook_output=False,
        pytest_args=[],
    )

    assert "--include-rollout-sweep" in steps[0].command


def test_main_refreshes_validation_snapshot_with_step_timings(monkeypatch):
    """Refresh the canonical validation snapshot after a successful E2E pass."""
    module = _load_run_e2e_module()
    steps = [
        module.CommandStep("pipeline", [module.sys.executable, "run_pipeline.py"]),
        module.CommandStep("notebooks", [module.sys.executable, "scripts/validate_notebooks.py"]),
        module.CommandStep("pytest", [module.sys.executable, "-m", "pytest", "-q"]),
    ]
    step_timings = {"pipeline": 12.5, "notebooks": 3.25, "pytest": 7.75}
    seen_steps: list[str] = []
    snapshot_call: dict[str, object] = {}
    visualization_call: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            mode="quick",
            without_performance=False,
            with_multires=True,
            with_rollout=True,
            with_rollout_sweep=True,
            with_horizon_curve=True,
            with_forecast_control=True,
            keep_notebook_output=False,
            pytest_arg=[],
        ),
    )
    monkeypatch.setattr(module, "build_command_plan", lambda **_: steps)
    monkeypatch.setattr(
        module,
        "_run_step",
        lambda step: seen_steps.append(step.name) or step_timings[step.name],
    )
    monkeypatch.setattr(
        module,
        "_write_validation_snapshot",
        lambda passed_step_seconds: (
            snapshot_call.setdefault("step_seconds", dict(passed_step_seconds)),
            module.PROJECT_ROOT / "docs" / "003_modeling" / "current_validation_snapshot.md",
        )[1],
    )
    monkeypatch.setattr(
        module,
        "_write_visualization_report",
        lambda: (
            visualization_call.setdefault("called", True),
            (
                module.PROJECT_ROOT / "docs" / "003_modeling" / "current_visualization_guide.md",
                module.PROJECT_ROOT / "outputs" / "reports" / "commercial_facility" / "latest" / "validation_dashboard.html",
            ),
        )[1],
    )

    result = module.main()

    assert result == 0
    assert seen_steps == ["pipeline", "notebooks", "pytest"]
    assert snapshot_call["step_seconds"] == step_timings
    assert visualization_call["called"] is True
