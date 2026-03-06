"""Pipeline orchestrator for raw -> bronze -> silver -> gold.

Purpose:
- Provide a single CLI entrypoint for stage execution and dry-run validation.
- Enforce startup configuration validation before any stage runs.
- Standardize structured console/file logging for operational traceability.

Last reviewed: 2026-03-06
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import PATHS, PROJECT_ROOT as CONFIG_PROJECT_ROOT, RESOLUTION_TO_SUFFIX
from config import DEFAULT_RESOLUTIONS, RESOLUTION_ALIASES, SUPPORTED_RESOLUTIONS
from config import validate_config
from utils import emit_quality_gate

logger = logging.getLogger("pipeline")


def _project_scoped_path(path: Path) -> Path:
    """Map config-resolved repo paths onto the active PROJECT_ROOT when needed."""
    if not path.is_absolute():
        return (PROJECT_ROOT / path).resolve()
    try:
        relative = path.relative_to(CONFIG_PROJECT_ROOT)
    except ValueError:
        return path
    return (PROJECT_ROOT / relative).resolve()


def _resolve_log_file_path() -> Path | None:
    """Resolve optional pipeline file log path.

    Environment override:
    - Unset: use `PATHS["logs_dir"] / "pipeline.log"`.
    - `ELF_PIPELINE_LOG_FILE=off|none|disable|disabled|0|false`: disable file logging.
    - Any other value: use that path (relative values are resolved from project root).
    """
    override = os.getenv("ELF_PIPELINE_LOG_FILE")
    if override is None or not override.strip():
        return _project_scoped_path(PATHS["logs_dir"]) / "pipeline.log"

    normalized = override.strip().lower()
    if normalized in {"off", "none", "disable", "disabled", "0", "false"}:
        return None

    path = Path(override).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _configure_logging(verbose: bool) -> None:
    """Configure console and file logging for pipeline execution."""
    log_level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)

    root.addHandler(console)

    log_file_path = _resolve_log_file_path()
    if log_file_path is not None:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        logfile = logging.FileHandler(log_file_path, encoding="utf-8")
        logfile.setLevel(log_level)
        logfile.setFormatter(formatter)
        root.addHandler(logfile)
    else:
        logger.debug("File logging disabled via ELF_PIPELINE_LOG_FILE")


def _load_module(path: Path, module_name: str) -> ModuleType:
    """Load a Python module from a file path using importlib."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_stage_function(stage: str) -> Callable[..., object]:
    """Resolve the callable entrypoint for a pipeline stage."""
    stage_modules = {
        "bronze": ("000_raw_to_bronze.py", "raw_to_bronze"),
        "silver": ("001_bronze_to_silver.py", "bronze_to_silver"),
        "gold": ("002_silver_to_gold.py", "silver_to_gold"),
    }
    filename, fn_name = stage_modules[stage]
    module = _load_module(SCRIPTS_DIR / filename, f"pipeline_{stage}")
    fn = getattr(module, fn_name, None)
    if fn is None:
        raise RuntimeError(f"Function '{fn_name}' not found in {filename}")
    return fn


def _normalize_paths(result: object) -> list[Path]:
    """Normalize stage return values into a list of output paths."""
    if result is None:
        return []
    if isinstance(result, Path):
        return [result]
    if isinstance(result, list):
        return [Path(p) for p in result]
    if isinstance(result, tuple):
        return [Path(p) for p in result]
    return []


def _validate_resolutions(resolution: str | None) -> list[str] | None:
    """Validate and canonicalize an optional resolution argument."""
    if resolution is None:
        return None
    canonical = RESOLUTION_ALIASES.get(resolution, resolution)
    if canonical not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"Unsupported resolution '{resolution}'. Supported: {SUPPORTED_RESOLUTIONS} "
            f"(aliases: {sorted(RESOLUTION_ALIASES)})"
        )
    return [canonical]


def _dry_run(stage: str, resolutions: list[str] | None) -> None:
    """Validate configuration, inputs, and output directories without transforms."""
    logger.info("Dry run validation started for stage='%s'", stage)
    validate_config()

    if stage in {"bronze", "all"}:
        raw_path = _project_scoped_path(PATHS["raw_mat"])
        if not raw_path.exists():
            raise ValueError(f"Missing required raw input: {raw_path}")

    if stage == "silver":
        bronze_path = _project_scoped_path(PATHS["bronze_file"])
        if not bronze_path.exists():
            raise ValueError(f"Missing required bronze input: {bronze_path}")

    if stage == "gold":
        target_resolutions = resolutions if resolutions is not None else list(DEFAULT_RESOLUTIONS)
        for res in target_resolutions:
            suffix = RESOLUTION_TO_SUFFIX[res]
            silver_path = _project_scoped_path(PATHS["silver_dir"]) / f"power_load_{suffix}.parquet"
            if not silver_path.exists():
                raise ValueError(f"Missing required silver input for {res}: {silver_path}")
    if stage == "performance":
        performance_script = SCRIPTS_DIR / "004_model_performance.py"
        if not performance_script.exists():
            raise ValueError(f"Missing required performance script: {performance_script}")
        manifest_path = _project_scoped_path(PATHS["outputs_modeling_dir"]) / "run_manifest.json"
        if not manifest_path.exists():
            for prerequisite in (
                SCRIPTS_DIR / "003_create_model_datasets.py",
                SCRIPTS_DIR / "validate_notebooks.py",
                PROJECT_ROOT / "notebooks" / "003_modeling.ipynb",
            ):
                if not prerequisite.exists():
                    raise ValueError(f"Missing required modeling prerequisite: {prerequisite}")

    for dir_key in ("logs_dir", "silver_dir", "gold_dir", "model_dir"):
        _project_scoped_path(PATHS[dir_key]).mkdir(parents=True, exist_ok=True)

    logger.info("Dry run validation completed successfully")


def _run_stage(stage: str, resolutions: list[str] | None) -> list[Path]:
    """Execute one stage and return normalized output file paths."""
    stage_fn = _load_stage_function(stage)
    start = time.perf_counter()
    logger.info("Stage '%s' started", stage)

    try:
        if stage in {"silver", "gold"} and resolutions is not None:
            outputs = _normalize_paths(stage_fn(resolutions=resolutions))
        else:
            outputs = _normalize_paths(stage_fn())
    except Exception as exc:
        elapsed = time.perf_counter() - start
        raise RuntimeError(f"Stage '{stage}' failed after {elapsed:.2f}s: {exc}") from exc

    elapsed = time.perf_counter() - start
    logger.info("Stage '%s' completed in %.2fs", stage, elapsed)

    for output in outputs:
        if output.exists():
            size_mb = output.stat().st_size / (1024 * 1024)
            logger.info("Stage '%s' output: %s (%.2f MB)", stage, output, size_mb)
        else:
            logger.warning("Stage '%s' reported output not found on disk: %s", stage, output)

    return outputs


def _run_performance_stage(
    resolutions: list[str] | None, *, performance_mode: str = "quick"
) -> list[Path]:
    """Execute stage-5 model performance script and return expected artifact paths."""
    script_path = SCRIPTS_DIR / "004_model_performance.py"
    if not script_path.exists():
        raise RuntimeError(f"Missing performance script: {script_path}")
    if performance_mode not in {"quick", "full", "preflight"}:
        raise ValueError(f"Unsupported performance mode: {performance_mode}")

    cmd = [sys.executable, str(script_path)]
    if resolutions:
        cmd.extend(["--resolution", resolutions[0]])
    if performance_mode == "quick":
        cmd.append("--quick")
    elif performance_mode == "preflight":
        cmd.append("--preflight-only")

    start = time.perf_counter()
    logger.info("Stage 'performance' started (mode=%s)", performance_mode)
    bootstrap_outputs = _ensure_step4_modeling_artifacts()
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        elapsed = time.perf_counter() - start
        raise RuntimeError(f"Stage 'performance' failed after {elapsed:.2f}s: {exc}") from exc
    elapsed = time.perf_counter() - start
    logger.info("Stage 'performance' completed in %.2fs", elapsed)
    return bootstrap_outputs + [
        _project_scoped_path(PATHS["outputs_performance_dir"]) / "run_manifest.json"
    ]


def _run_modeling_stage() -> list[Path]:
    """Generate model datasets and execute the modeling notebook artifact export."""
    dataset_script = SCRIPTS_DIR / "003_create_model_datasets.py"
    notebook_runner = SCRIPTS_DIR / "validate_notebooks.py"
    modeling_notebook = PROJECT_ROOT / "notebooks" / "003_modeling.ipynb"

    for path in (dataset_script, notebook_runner, modeling_notebook):
        if not path.exists():
            raise RuntimeError(f"Missing modeling prerequisite: {path}")

    start = time.perf_counter()
    logger.info("Stage 'modeling' started (bootstrapping step-4 artifacts)")
    try:
        subprocess.run([sys.executable, str(dataset_script)], cwd=PROJECT_ROOT, check=True)
        subprocess.run(
            [sys.executable, str(notebook_runner), "--notebook", str(modeling_notebook)],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        elapsed = time.perf_counter() - start
        raise RuntimeError(f"Stage 'modeling' failed after {elapsed:.2f}s: {exc}") from exc

    elapsed = time.perf_counter() - start
    logger.info("Stage 'modeling' completed in %.2fs", elapsed)
    return [_project_scoped_path(PATHS["outputs_modeling_dir"]) / "run_manifest.json"]


def _ensure_step4_modeling_artifacts() -> list[Path]:
    """Ensure step-4 modeling artifacts exist before stage-5 evaluation runs."""
    manifest_path = _project_scoped_path(PATHS["outputs_modeling_dir"]) / "run_manifest.json"
    if manifest_path.exists():
        return [manifest_path]
    logger.info(
        "Stage-4 modeling artifacts are missing; running modeling prerequisites before performance."
    )
    return _run_modeling_stage()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for pipeline orchestration."""
    parser = argparse.ArgumentParser(description="Run the electric load forecasting data pipeline.")
    parser.add_argument(
        "--stage",
        choices=["all", "bronze", "silver", "gold", "performance"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--resolution",
        choices=sorted(set(SUPPORTED_RESOLUTIONS) | set(RESOLUTION_ALIASES)),
        default=None,
        help="Limit silver/gold runs to a single resolution.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and required inputs without running transformations.",
    )
    parser.add_argument(
        "--include-performance",
        action="store_true",
        help="When stage=all, also run stage-5 performance evaluation.",
    )
    parser.add_argument(
        "--performance-mode",
        choices=["quick", "full", "preflight"],
        default="quick",
        help="Execution mode for performance stage (stage=performance or --include-performance).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the configured pipeline stage(s) and return process exit code."""
    args = parse_args()
    _configure_logging(verbose=args.verbose)
    requested_stages: list[str] = []
    completed_stages: list[str] = []

    try:
        validate_config()
        resolutions = _validate_resolutions(args.resolution)
    except Exception as exc:
        logger.error("Resolution validation failed: %s", exc)
        return 2

    try:
        if args.dry_run:
            _dry_run(args.stage, resolutions)
            return 0

        pipeline_start = time.perf_counter()
        outputs: list[Path] = []

        if args.stage == "all":
            requested_stages = ["bronze", "silver", "gold"]
            if args.include_performance:
                requested_stages.append("performance")
        else:
            requested_stages = [args.stage]

        if args.stage == "all":
            outputs.extend(_run_stage("bronze", None))
            completed_stages.append("bronze")
            outputs.extend(_run_stage("silver", resolutions))
            completed_stages.append("silver")
            outputs.extend(_run_stage("gold", resolutions))
            completed_stages.append("gold")
            if args.include_performance:
                outputs.extend(_run_performance_stage(resolutions, performance_mode=args.performance_mode))
                completed_stages.append("performance")
        elif args.stage == "performance":
            outputs.extend(_run_performance_stage(resolutions, performance_mode=args.performance_mode))
            completed_stages.append("performance")
        else:
            outputs.extend(_run_stage(args.stage, resolutions))
            completed_stages.append(args.stage)

        total_elapsed = time.perf_counter() - pipeline_start
        emit_quality_gate(
            "PIPELINE HEALTH",
            len(completed_stages) == len(requested_stages),
            details={
                "completed": f"{len(completed_stages)}/{len(requested_stages)}",
                "stages": ",".join(requested_stages),
            },
            logger_instance=logger,
        )
        logger.info("Pipeline completed successfully in %.2fs", total_elapsed)
        if outputs:
            logger.info("Total output files generated: %d", len(outputs))
        return 0
    except Exception as exc:
        if "requested_stages" in locals():
            emit_quality_gate(
                "PIPELINE HEALTH",
                False,
                details={
                    "completed": f"{len(completed_stages)}/{len(requested_stages)}",
                    "stages": ",".join(requested_stages),
                },
                logger_instance=logger,
                failure_level=logging.ERROR,
            )
        logger.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

