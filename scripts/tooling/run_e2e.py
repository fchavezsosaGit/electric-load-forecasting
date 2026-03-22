"""Run the repository verification flow from one stable command.

This tool is the operator-facing shortcut for the most common validation path:

- execute the configured pipeline surface
- validate and archive notebooks
- run pytest
- refresh the canonical current-validation snapshot from the latest artifacts

It is intentionally lightweight. The real stage behavior lives elsewhere; this
module only assembles a repeatable command plan, reports timings, and refreshes
the one-page current-state summary after a successful pass.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLING_DIR = Path(__file__).resolve().parent
if str(TOOLING_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLING_DIR))

from write_validation_snapshot import write_validation_snapshot
from write_visualization_report import write_visualization_report


@dataclass(frozen=True)
class CommandStep:
    """A named subprocess invocation used by the end-to-end runner."""

    name: str
    command: list[str]


def build_command_plan(
    *,
    mode: str,
    include_performance: bool,
    include_multires: bool,
    include_rollout: bool,
    include_rollout_sweep: bool,
    include_horizon_curve: bool,
    include_forecast_control: bool,
    keep_notebook_output: bool,
    pytest_args: list[str],
) -> list[CommandStep]:
    """Build the ordered subprocess plan for one E2E invocation.

    The returned steps are intentionally explicit because they are surfaced in
    logs and timing summaries. This function is used by the CLI entrypoint and
    by tests that verify the repository's operational command contract.
    """
    performance_mode = "full" if mode == "full" else "quick"
    pipeline_command = [sys.executable, "run_pipeline.py", "--stage", "all"]
    if include_performance:
        pipeline_command.extend(["--include-performance", "--performance-mode", performance_mode])
    if include_multires:
        pipeline_command.extend(["--include-multires", "--multires-mode", "candidate" if mode == "full" else "smoke"])
    if include_rollout:
        pipeline_command.append("--include-rollout")
    if include_rollout_sweep:
        pipeline_command.append("--include-rollout-sweep")
    if include_horizon_curve:
        pipeline_command.append("--include-horizon-curve")
    if include_forecast_control:
        pipeline_command.append("--include-forecast-control")

    notebook_command = [sys.executable, "scripts/validate_notebooks.py"]
    if keep_notebook_output:
        notebook_command.append("--keep-output")

    pytest_command = [sys.executable, "-m", "pytest", "-q", *pytest_args]
    return [
        CommandStep("pipeline", pipeline_command),
        CommandStep("notebooks", notebook_command),
        CommandStep("pytest", pytest_command),
    ]


def _run_step(step: CommandStep) -> float:
    """Execute one E2E subprocess step and return its wall-clock runtime."""
    print(f"[run_e2e] starting {step.name}: {' '.join(step.command)}", flush=True)
    start = time.perf_counter()
    subprocess.run(step.command, cwd=PROJECT_ROOT, check=True)
    return time.perf_counter() - start


def _write_validation_snapshot(step_seconds: dict[str, float]) -> Path:
    """Refresh the canonical current-validation snapshot after a successful E2E run."""
    return write_validation_snapshot(step_seconds=step_seconds)


def _write_visualization_report() -> tuple[Path, Path]:
    """Refresh the integrated latest-state visualization guide and dashboard."""
    result = write_visualization_report()
    return (result.guide_path, result.dashboard_path)


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the repository E2E runner."""
    parser = argparse.ArgumentParser(description="Run pipeline, notebooks, and tests as one E2E workflow.")
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument(
        "--without-performance",
        action="store_true",
        help="Skip Stage-5 performance execution inside the pipeline step.",
    )
    parser.add_argument(
        "--with-multires",
        action="store_true",
        help="Include stage-6 multiresolution comparison inside the pipeline step.",
    )
    parser.add_argument(
        "--with-rollout",
        action="store_true",
        help="Include stage-7 recursive rollout inside the pipeline step.",
    )
    parser.add_argument(
        "--with-rollout-sweep",
        action="store_true",
        help="Include the Stage-7 challenger-sweep surface inside the pipeline step.",
    )
    parser.add_argument(
        "--with-horizon-curve",
        action="store_true",
        help="Include the H5 horizon-characterization stage inside the pipeline step.",
    )
    parser.add_argument(
        "--with-forecast-control",
        action="store_true",
        help="Include the Stage-10 forecast-control backtest inside the pipeline step.",
    )
    parser.add_argument(
        "--keep-notebook-output",
        action="store_true",
        help="Retain notebook cell outputs instead of clearing transient execution output.",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Additional argument to forward to pytest. Provide multiple times as needed.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute the E2E command plan and print per-step timings."""
    args = parse_args()
    steps = build_command_plan(
        mode=args.mode,
        include_performance=not args.without_performance,
        include_multires=bool(args.with_multires),
        include_rollout=bool(args.with_rollout),
        include_rollout_sweep=bool(args.with_rollout_sweep),
        include_horizon_curve=bool(args.with_horizon_curve),
        include_forecast_control=bool(args.with_forecast_control),
        keep_notebook_output=bool(args.keep_notebook_output),
        pytest_args=list(args.pytest_arg),
    )

    total_start = time.perf_counter()
    step_seconds: dict[str, float] = {}
    for step in steps:
        elapsed = _run_step(step)
        step_seconds[step.name] = elapsed
        print(f"[run_e2e] {step.name} completed in {elapsed:.2f}s")
    total_elapsed = time.perf_counter() - total_start
    snapshot_path = _write_validation_snapshot(step_seconds)
    visualization_guide_path, dashboard_path = _write_visualization_report()
    print(f"[run_e2e] validation snapshot refreshed at {snapshot_path}")
    print(f"[run_e2e] visualization guide refreshed at {visualization_guide_path}")
    print(f"[run_e2e] dashboard refreshed at {dashboard_path}")
    print(f"[run_e2e] all steps completed in {total_elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
