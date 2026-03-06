"""Run the full repository verification flow from a single entrypoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CommandStep:
    """A named subprocess invocation used by the end-to-end runner."""

    name: str
    command: list[str]


def build_command_plan(
    *,
    mode: str,
    include_performance: bool,
    keep_notebook_output: bool,
    pytest_args: list[str],
) -> list[CommandStep]:
    performance_mode = "full" if mode == "full" else "quick"
    pipeline_command = [sys.executable, "run_pipeline.py", "--stage", "all"]
    if include_performance:
        pipeline_command.extend(["--include-performance", "--performance-mode", performance_mode])

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
    print(f"[run_e2e] starting {step.name}: {' '.join(step.command)}", flush=True)
    start = time.perf_counter()
    subprocess.run(step.command, cwd=PROJECT_ROOT, check=True)
    return time.perf_counter() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pipeline, notebooks, and tests as one E2E workflow.")
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument(
        "--without-performance",
        action="store_true",
        help="Skip Stage-5 performance execution inside the pipeline step.",
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
    args = parse_args()
    steps = build_command_plan(
        mode=args.mode,
        include_performance=not args.without_performance,
        keep_notebook_output=bool(args.keep_notebook_output),
        pytest_args=list(args.pytest_arg),
    )

    total_start = time.perf_counter()
    for step in steps:
        elapsed = _run_step(step)
        print(f"[run_e2e] {step.name} completed in {elapsed:.2f}s")
    total_elapsed = time.perf_counter() - total_start
    print(f"[run_e2e] all steps completed in {total_elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
