"""Execute project notebooks end to end as a reproducible validation step.

This runner is used by both direct notebook smoke checks and the repository E2E
flow. It does three important jobs:

- execute the core notebooks under controlled settings
- refresh the Stage-4 modeling dependencies before `003_modeling.ipynb`
- archive executed notebooks and validate the Stage-4 figure and metric outputs

The archive and artifact-validation surface exists so reviewers can answer the
common questions "what ran?" and "what visual evidence was produced?" from one
timestamped output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import DATASET, PATHS, preferred_output_path, scoped_output_path
from modeling.common import FigureGuideEntry, validate_png_artifact, write_figure_guide

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_RUNS_DIR = scoped_output_path(PROJECT_ROOT / "outputs" / "008_notebook_runs")
DEFAULT_NOTEBOOKS = [
    NOTEBOOK_DIR / "000_raw_eda.ipynb",
    NOTEBOOK_DIR / "001_bronze_eda.ipynb",
    NOTEBOOK_DIR / "002_silver_eda.ipynb",
    NOTEBOOK_DIR / "003_modeling.ipynb",
]
logger = logging.getLogger(__name__)
SILVER_NOTEBOOK_NAME = "002_silver_eda.ipynb"
MODELING_NOTEBOOK_NAME = "003_modeling.ipynb"
MODELING_REQUIRED_METRIC_COLUMNS = {
    "metrics_overall.csv": {"mae", "mae_pct", "rmse", "rmse_pct"},
    "metrics_by_day_class.csv": {"mae", "mae_pct", "rmse", "rmse_pct"},
    "metrics_by_hour.csv": {"mae", "mae_pct", "rmse", "rmse_pct"},
}
MODELING_REQUIRED_PNGS = (
    "fig_actual_vs_predicted.png",
    "fig_error_by_hour.png",
    "fig_model_comparison.png",
    "fig_day_ahead.png",
)
MODELING_FIGURE_GUIDE = {
    "fig_actual_vs_predicted.png": FigureGuideEntry(
        filename="fig_actual_vs_predicted.png",
        title="Actual vs predicted overlay",
        intent="Show whether the selected validation-day forecast follows the observed load shape and turning points.",
        how_to_read="Compare the learned curve and baselines against the actual load line across the same timestamps.",
        look_for="Large misses at ramps, sustained bias above or below the actual curve, and whether the chosen model improves on persistence where operations change quickly.",
    ),
    "fig_error_by_hour.png": FigureGuideEntry(
        filename="fig_error_by_hour.png",
        title="MAE by hour of day",
        intent="Show when during the day the forecast family is weakest or strongest.",
        how_to_read="Read each line as the average absolute error for that model family at each clock hour.",
        look_for="Error spikes during morning start-up, lunch transitions, or evening shut-down; these reveal where feature design or baseline corrections still need work.",
    ),
    "fig_model_comparison.png": FigureGuideEntry(
        filename="fig_model_comparison.png",
        title="Model comparison summary",
        intent="Give a compact benchmark comparison across the Stage-4 experiment grid and baselines.",
        how_to_read="Compare bar heights and annotated labels across models, feature sets, and baseline rows.",
        look_for="Whether learned models beat persistence on the selected metric, and whether any win depends on low coverage or unstable complexity.",
    ),
    "fig_day_ahead.png": FigureGuideEntry(
        filename="fig_day_ahead.png",
        title="Day-ahead profile example",
        intent="Show the 24-hour shape quality of the day-ahead extension against actual load and prior-day structure.",
        how_to_read="Track whether the predicted profile captures the right daily envelope, peak timing, and trough timing even when pointwise error remains high.",
        look_for="Profile-shape alignment, missed peak timing, and whether the forecast is useful as an operational planning surface before intraday corrections arrive.",
    ),
}
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


def _resolve_notebook_path(path: Path) -> Path:
    """Resolve notebook paths relative to the project root when needed."""
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _serialize_source_path(path: Path) -> str:
    """Serialize source notebook paths relative to the repo when possible."""
    resolved_path = _resolve_notebook_path(path)
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return resolved_path.as_posix()


def execute_notebook(path: Path, env_overrides: dict[str, str] | None = None) -> None:
    """Execute one notebook in place via nbconvert with Windows-safe event loop policy."""
    resolved_path = _resolve_notebook_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Notebook not found: {resolved_path}")

    cmd = _build_nbconvert_command(resolved_path)
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Notebook execution failed for {resolved_path}: {exc}") from exc


def _run_repo_script(*, relative_path: str, failure_message: str) -> None:
    """Run one repo script from the project root with consistent error handling."""
    script_path = PROJECT_ROOT / relative_path
    if not script_path.exists():
        raise FileNotFoundError(f"Required script not found: {script_path}")
    try:
        subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{failure_message}: {exc}") from exc


def _refresh_modeling_inputs() -> None:
    """Refresh silver, gold, and model datasets before executing the modeling notebook."""
    _run_repo_script(
        relative_path="scripts/001_bronze_to_silver.py",
        failure_message="Silver refresh failed",
    )
    _run_repo_script(
        relative_path="scripts/002_silver_to_gold.py",
        failure_message="Gold refresh failed",
    )
    _run_repo_script(
        relative_path="scripts/003_create_model_datasets.py",
        failure_message="Model dataset refresh failed",
    )


def _resolve_notebook_runs_dir() -> Path:
    """Resolve the root used to archive executed notebook snapshots."""
    override = os.getenv("ELF_NOTEBOOK_RUNS_DIR")
    if override and override.strip():
        root = Path(override).expanduser()
    else:
        root = NOTEBOOK_RUNS_DIR
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    return root


def _prepare_archive_run_dir(output_root: Path) -> Path:
    """Create one timestamped notebook archive run directory."""
    run_dir = output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _archive_label(nb: Path, *, profile_index: int | None = None) -> str:
    """Build a stable archive label for the default notebook run or a profile run."""
    if profile_index is None:
        return nb.stem
    return f"{nb.stem}__profile_{profile_index:02d}"


def _summarize_notebook_payload(path: Path) -> dict[str, int | None]:
    """Return lightweight notebook execution summary metadata."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "code_cell_count": None,
            "executed_code_cell_count": None,
            "output_count": None,
            "error_output_count": None,
        }

    cells = payload.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    output_count = 0
    error_output_count = 0
    executed_count = 0
    for cell in code_cells:
        if cell.get("execution_count") is not None:
            executed_count += 1
        outputs = cell.get("outputs", [])
        output_count += len(outputs)
        error_output_count += sum(1 for item in outputs if item.get("output_type") == "error")
    return {
        "code_cell_count": len(code_cells),
        "executed_code_cell_count": executed_count,
        "output_count": output_count,
        "error_output_count": error_output_count,
    }


def _archive_notebook_snapshot(
    *,
    notebook_path: Path,
    run_dir: Path,
    env_overrides: dict[str, str] | None,
    profile_index: int | None,
    clear_outputs: bool,
) -> dict[str, Any]:
    """Archive one executed notebook before tracked outputs are cleared."""
    resolved_path = _resolve_notebook_path(notebook_path)
    archive_dir = run_dir / "notebooks"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"{_archive_label(notebook_path, profile_index=profile_index)}.ipynb"
    archive_path = archive_dir / archive_name
    shutil.copy2(resolved_path, archive_path)
    summary = _summarize_notebook_payload(archive_path)
    return {
        "source_path": _serialize_source_path(notebook_path),
        "archive_path": str(archive_path.relative_to(run_dir)).replace("\\", "/"),
        "profile_index": profile_index,
        "env_overrides": dict(env_overrides or {}),
        "clear_outputs_after_archive": bool(clear_outputs),
        "file_size_bytes": int(archive_path.stat().st_size),
        **summary,
    }


def _validate_modeling_outputs() -> dict[str, Any]:
    """Validate Stage-4 modeling artifacts produced by the notebook."""
    output_dir = preferred_output_path(PATHS["outputs_modeling_dir"])
    if not output_dir.exists():
        raise FileNotFoundError(f"Modeling output directory not found: {output_dir}")

    csv_summary: dict[str, Any] = {}
    for filename, required_columns in MODELING_REQUIRED_METRIC_COLUMNS.items():
        path = output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required modeling artifact: {path}")
        frame = pd.read_csv(path)
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise ValueError(
                f"Modeling artifact {path} is missing required columns: {missing_columns}"
            )
        csv_summary[filename] = {
            "rows": int(len(frame)),
            "required_columns_present": True,
        }

    png_summary: dict[str, Any] = {}
    for filename in MODELING_REQUIRED_PNGS:
        path = output_dir / filename
        width, height = validate_png_artifact(path)
        guide = MODELING_FIGURE_GUIDE[filename]
        png_summary[filename] = {
            "width": width,
            "height": height,
            "title": guide.title,
            "intent": guide.intent,
            "how_to_read": guide.how_to_read,
            "look_for": guide.look_for,
        }

    write_figure_guide(
        output_path=output_dir / "figure_guide.md",
        stage_title="Stage-4 Modeling Figures",
        stage_purpose=(
            "These notebook-produced figures are the primary visual evidence for the "
            "Stage-4 benchmark surface. They explain how the current `1min` modeling "
            "stack is measured and where reviewers should expect forecast quality to "
            "succeed or fail."
        ),
        figures=[MODELING_FIGURE_GUIDE[name] for name in MODELING_REQUIRED_PNGS],
    )

    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "load_type": manifest.get("load_type"),
        "artifact_namespace": manifest.get("artifact_namespace"),
        "metric_percentage_basis": manifest.get("metric_percentage_basis"),
        "csv_artifacts": csv_summary,
        "png_artifacts": png_summary,
        "figure_guide": "figure_guide.md",
    }


def _write_archive_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    """Persist notebook archive manifest for one validation invocation."""
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _refresh_latest_archive(run_dir: Path, output_root: Path) -> None:
    """Refresh a convenience copy of the latest successful notebook archive run."""
    latest_dir = output_root / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)


def _clear_notebook_outputs(path: Path) -> None:
    """Remove transient execution outputs while preserving source and counts."""
    resolved_path = _resolve_notebook_path(path)
    notebook = nbformat.read(resolved_path, as_version=4)
    changed = False
    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
    if changed:
        nbformat.write(notebook, resolved_path)


def validate_notebooks(notebooks: list[Path], *, clear_outputs: bool = True) -> None:
    """Execute all provided notebooks and raise on first failure."""
    output_root = _resolve_notebook_runs_dir()
    run_dir = _prepare_archive_run_dir(output_root)
    manifest: dict[str, Any] = {
        "run_id": run_dir.name,
        "stage": "008_notebook_runs",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "load_type": DATASET["load_type"],
        "artifact_namespace": DATASET["artifact_namespace"],
        "clear_outputs": bool(clear_outputs),
        "status": "running",
        "notebooks": [],
        "model_datasets_refreshed": False,
        "warnings": [],
    }

    try:
        for nb in notebooks:
            if nb.name == MODELING_NOTEBOOK_NAME:
                logger.info(
                    "Refreshing silver, gold, and model datasets before modeling notebook validation."
                )
                _refresh_modeling_inputs()
                manifest["model_datasets_refreshed"] = True

            logger.info("Executing notebook: %s", nb)
            execute_notebook(nb)
            notebook_entry = _archive_notebook_snapshot(
                notebook_path=nb,
                run_dir=run_dir,
                env_overrides=None,
                profile_index=None,
                clear_outputs=clear_outputs,
            )
            if nb.name == MODELING_NOTEBOOK_NAME:
                notebook_entry["artifact_validation"] = _validate_modeling_outputs()
            manifest["notebooks"].append(notebook_entry)
            if clear_outputs:
                _clear_notebook_outputs(nb)

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
                    manifest["notebooks"].append(
                        _archive_notebook_snapshot(
                            notebook_path=nb,
                            run_dir=run_dir,
                            env_overrides=profile,
                            profile_index=idx,
                            clear_outputs=clear_outputs,
                        )
                    )
                    if clear_outputs:
                        _clear_notebook_outputs(nb)
        manifest["status"] = "success"
        _write_archive_manifest(run_dir, manifest)
        _refresh_latest_archive(run_dir, output_root)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        _write_archive_manifest(run_dir, manifest)
        raise

    logger.info("Notebook validation completed successfully. archive=%s", run_dir)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for notebook validation."""
    parser = argparse.ArgumentParser(description="Execute project notebooks for smoke validation.")
    parser.add_argument(
        "--notebook",
        action="append",
        help="Path to a notebook to execute. Can be provided multiple times.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Retain cell outputs after execution instead of clearing transient notebook output.",
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
        validate_notebooks(notebooks, clear_outputs=not args.keep_output)
        return 0
    except Exception as exc:
        logger.error("Notebook validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
