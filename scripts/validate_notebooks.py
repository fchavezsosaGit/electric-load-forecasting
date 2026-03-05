"""Execute project notebooks end-to-end as a smoke validation step.

For the silver notebook, run additional validation profiles covering
all resolution modes and mixed AUTO-flag settings.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
DEFAULT_NOTEBOOKS = [
    NOTEBOOK_DIR / "000_raw_eda.ipynb",
    NOTEBOOK_DIR / "001_bronze_eda.ipynb",
    NOTEBOOK_DIR / "002_silver_eda.ipynb",
    NOTEBOOK_DIR / "003_modeling.ipynb",
]
logger = logging.getLogger(__name__)
SILVER_NOTEBOOK_NAME = "002_silver_eda.ipynb"
SILVER_VALIDATION_PROFILES = [
    {
        "ELF_NB_RESOLUTION_MODE": "default",
        "ELF_NB_AUTO_BINS": "true",
        "ELF_NB_AUTO_ACF_DEPTH": "true",
    },
    {
        "ELF_NB_RESOLUTION_MODE": "all",
        "ELF_NB_AUTO_BINS": "false",
        "ELF_NB_AUTO_ACF_DEPTH": "false",
    },
    {
        "ELF_NB_RESOLUTION_MODE": "custom",
        "ELF_NB_CUSTOM_RESOLUTIONS": "5min,15min",
        "ELF_NB_AUTO_BINS": "true",
        "ELF_NB_AUTO_ACF_DEPTH": "false",
    },
]

_NBCONVERT_RUNNER = """
import asyncio
import sys
from nbconvert.nbconvertapp import main as nbconvert_main

if sys.platform.startswith("win") and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

notebook_path = sys.argv[1]
sys.argv = ["jupyter-nbconvert", "--to", "notebook", "--execute", "--inplace", notebook_path]
nbconvert_main()
""".strip()


def _configure_logging() -> None:
    """Configure default logging for notebook validation runs."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _build_nbconvert_command(path: Path) -> list[str]:
    """Build a Python-managed nbconvert command for deterministic cross-platform execution."""
    return [sys.executable, "-c", _NBCONVERT_RUNNER, str(path)]


def execute_notebook(path: Path, env_overrides: dict[str, str] | None = None) -> None:
    """Execute one notebook in place via nbconvert with Windows-safe event loop policy."""
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")

    cmd = _build_nbconvert_command(path)
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Notebook execution failed for {path}: {exc}") from exc


def validate_notebooks(notebooks: list[Path]) -> None:
    """Execute all provided notebooks and raise on first failure."""
    for nb in notebooks:
        logger.info("Executing notebook: %s", nb)
        execute_notebook(nb)
        if nb.name == SILVER_NOTEBOOK_NAME:
            for idx, profile in enumerate(SILVER_VALIDATION_PROFILES, start=1):
                logger.info(
                    "Executing silver validation profile %d/%d: mode=%s, auto_bins=%s, auto_acf_depth=%s",
                    idx,
                    len(SILVER_VALIDATION_PROFILES),
                    profile.get("ELF_NB_RESOLUTION_MODE"),
                    profile.get("ELF_NB_AUTO_BINS"),
                    profile.get("ELF_NB_AUTO_ACF_DEPTH"),
                )
                execute_notebook(nb, env_overrides=profile)
    logger.info("Notebook validation completed successfully.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for notebook validation."""
    parser = argparse.ArgumentParser(description="Execute project notebooks for smoke validation.")
    parser.add_argument(
        "--notebook",
        action="append",
        help="Path to a notebook to execute. Can be provided multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    """Run notebook validation and return process exit code."""
    _configure_logging()
    args = parse_args()
    if args.notebook:
        notebooks = [Path(p).resolve() for p in args.notebook]
    else:
        notebooks = DEFAULT_NOTEBOOKS

    try:
        validate_notebooks(notebooks)
        return 0
    except Exception as exc:
        logger.error("Notebook validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
