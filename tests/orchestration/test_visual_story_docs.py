"""Notebook and markdown visual-story consistency tests."""

from __future__ import annotations

import json
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Find the repository root so the doc fixtures can be loaded reliably."""
    for candidate in [start, *start.parents]:
        if (candidate / "run_pipeline.py").exists() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Unable to locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)


def test_modeling_notebook_contains_cross_stage_visual_story():
    """The canonical modeling notebook should include curated visual-reading anchors."""
    notebook_path = PROJECT_ROOT / "notebooks" / "003_modeling.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    )

    assert "## Results Assembly, Decision Layers, and Determinism" in markdown
    assert "## Cross-Stage Visual Anchors for Markdown Surfaces" in markdown
    assert "../outputs/005_performance/commercial_facility/latest/fig_holdout_benchmark_ci.png" in markdown
    assert "../outputs/009_horizon_curve/commercial_facility/latest/fig_horizon_ratio_curve.png" in markdown
    assert "../outputs/010_forecast_control/commercial_facility/latest/fig_control_layer_gain_ci.png" in markdown


def test_report_iv_run_summary_links_current_visual_surfaces():
    """The long-form run summary should point readers to the maintained visual surfaces."""
    summary_path = PROJECT_ROOT / "docs" / "003_modeling" / "report_iv_run_summary.md"
    content = summary_path.read_text(encoding="utf-8")

    assert "## Recommended Visual Anchors" in content
    assert "[`current_visualization_guide.md`](current_visualization_guide.md)" in content
    assert "[validation dashboard](../../outputs/reports/commercial_facility/latest/validation_dashboard.html)" in content
    assert "../../outputs/005_performance/commercial_facility/latest/fig_holdout_benchmark_ci.png" in content
