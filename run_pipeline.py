"""Pipeline orchestrator for raw -> bronze -> silver -> gold.

Purpose:
- Provide a single CLI entrypoint for stage execution and dry-run validation.
- Enforce startup configuration validation before any stage runs.
- Standardize structured console/file logging for operational traceability.

Last reviewed: 2026-02-20
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import PATHS, RESOLUTION_TO_SUFFIX
from config import DEFAULT_RESOLUTIONS, RESOLUTION_ALIASES, SUPPORTED_RESOLUTIONS
from config import validate_config

logger = logging.getLogger("pipeline")


def _resolve_log_file_path() -> Path | None:
    """Resolve optional pipeline file log path.

    Environment override:
    - Unset: use `PATHS["logs_dir"] / "pipeline.log"`.
    - `ELF_PIPELINE_LOG_FILE=off|none|disable|disabled|0|false`: disable file logging.
    - Any other value: use that path (relative values are resolved from project root).
    """
    override = os.getenv("ELF_PIPELINE_LOG_FILE")
    if override is None or not override.strip():
        return PATHS["logs_dir"] / "pipeline.log"

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
        if not PATHS["raw_mat"].exists():
            raise ValueError(f"Missing required raw input: {PATHS['raw_mat']}")

    if stage == "silver":
        if not PATHS["bronze_file"].exists():
            raise ValueError(f"Missing required bronze input: {PATHS['bronze_file']}")

    if stage == "gold":
        target_resolutions = resolutions if resolutions is not None else list(DEFAULT_RESOLUTIONS)
        for res in target_resolutions:
            suffix = RESOLUTION_TO_SUFFIX[res]
            silver_path = PATHS["silver_dir"] / f"power_load_{suffix}.parquet"
            if not silver_path.exists():
                raise ValueError(f"Missing required silver input for {res}: {silver_path}")

    for dir_key in ("logs_dir", "silver_dir", "gold_dir", "model_dir"):
        PATHS[dir_key].mkdir(parents=True, exist_ok=True)

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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for pipeline orchestration."""
    parser = argparse.ArgumentParser(description="Run the electric load forecasting data pipeline.")
    parser.add_argument(
        "--stage",
        choices=["all", "bronze", "silver", "gold"],
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
    return parser.parse_args()


def main() -> int:
    """Run the configured pipeline stage(s) and return process exit code."""
    args = parse_args()
    _configure_logging(verbose=args.verbose)

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
            outputs.extend(_run_stage("bronze", None))
            outputs.extend(_run_stage("silver", resolutions))
            outputs.extend(_run_stage("gold", resolutions))
        else:
            outputs.extend(_run_stage(args.stage, resolutions))

        total_elapsed = time.perf_counter() - pipeline_start
        logger.info("Pipeline completed successfully in %.2fs", total_elapsed)
        if outputs:
            logger.info("Total output files generated: %d", len(outputs))
        return 0
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

