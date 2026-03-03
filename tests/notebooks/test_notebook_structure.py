"""Notebook structure checks for reproducible EDA artifacts."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOKS = [
    Path("notebooks/000_raw_eda.ipynb"),
    Path("notebooks/001_bronze_eda.ipynb"),
    Path("notebooks/002_silver_eda.ipynb"),
]


def _cell_executed(cell: dict[str, object]) -> bool:
    """Return True when a code cell shows execution evidence.

    Some nbconvert/jupyter combinations persist execution timestamps in cell metadata
    but omit numeric execution_count for certain cells. Treat either signal as executed.
    """
    if cell.get("execution_count") is not None:
        return True
    metadata = cell.get("metadata", {})
    if isinstance(metadata, dict):
        execution = metadata.get("execution")
        if isinstance(execution, dict):
            return "iopub.status.idle" in execution
    return False


def test_notebooks_have_markdown_config_and_execution():
    """Ensure notebooks include narrative, config usage, and executed code cells."""
    for notebook_path in NOTEBOOKS:
        data = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = data.get("cells", [])
        markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        source_text = "\n".join("".join(c.get("source", [])) for c in cells)

        assert len(markdown_cells) >= 3, f"{notebook_path} must include narrative markdown cells."
        assert "from scripts.config import PATHS" in source_text, (
            f"{notebook_path} must use config-driven paths."
        )
        assert "[`scripts/config.py`](../scripts/config.py)" in source_text, (
            f"{notebook_path} must link to configuration source files."
        )
        assert code_cells, f"{notebook_path} must contain code cells."
        assert all(_cell_executed(c) for c in code_cells), (
            f"{notebook_path} contains unexecuted code cells."
        )
