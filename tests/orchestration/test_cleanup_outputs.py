"""Output cleanup planning tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Find the repository root so the cleanup tool can be imported from tests."""
    for candidate in [start, *start.parents]:
        if (candidate / "run_pipeline.py").exists() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Unable to locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "tooling"


def _load_cleanup_outputs_module():
    """Load the cleanup tool directly from disk for deterministic planning tests."""
    path = SCRIPTS_DIR / "cleanup_outputs.py"
    spec = importlib.util.spec_from_file_location("test_cleanup_outputs_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_find_referenced_run_ids_scans_nested_targets(tmp_path):
    """Collect run ids from files and directories that contain evidence references."""
    module = _load_cleanup_outputs_module()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "summary.md").write_text("keep 20260312T064827127387Z and 20260310T231235398730Z", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore no ids here", encoding="utf-8")

    run_ids = module.find_referenced_run_ids([docs, tmp_path / "notes.txt"])

    assert run_ids == {"20260312T064827127387Z", "20260310T231235398730Z"}


def test_plan_cleanup_keeps_recent_and_referenced_runs(tmp_path):
    """Preserve the recent buffer and explicitly referenced historical evidence."""
    module = _load_cleanup_outputs_module()
    root = tmp_path / "outputs"
    root.mkdir()
    for run_id in [
        "20260310T000000000000Z",
        "20260311T000000000000Z",
        "20260312T000000000000Z",
    ]:
        run_dir = root / run_id
        run_dir.mkdir()
        (run_dir / "metrics.csv").write_text("value\n1\n", encoding="utf-8")
    (root / "latest").mkdir()

    policy = module.CleanupRootPolicy(root=root, keep_recent=1, label="test")
    decisions = module.plan_cleanup([policy], {"20260310T000000000000Z"})
    by_run_id = {decision.run_id: decision for decision in decisions}

    assert by_run_id["20260312T000000000000Z"].action == "keep"
    assert "recent_buffer" in by_run_id["20260312T000000000000Z"].reasons
    assert by_run_id["20260310T000000000000Z"].action == "keep"
    assert "referenced_by_docs_or_latest" in by_run_id["20260310T000000000000Z"].reasons
    assert by_run_id["20260311T000000000000Z"].action == "delete"


def test_execute_cleanup_removes_only_planned_directories(tmp_path):
    """Delete only the runs marked for removal while leaving kept runs intact."""
    module = _load_cleanup_outputs_module()
    keep_dir = tmp_path / "20260312T000000000000Z"
    delete_dir = tmp_path / "20260311T000000000000Z"
    keep_dir.mkdir()
    delete_dir.mkdir()
    (keep_dir / "metrics.csv").write_text("value\n1\n", encoding="utf-8")
    (delete_dir / "metrics.csv").write_text("value\n1\n", encoding="utf-8")

    decisions = [
        module.CleanupDecision(
            root=tmp_path,
            run_id=keep_dir.name,
            path=keep_dir,
            action="keep",
            reasons=("recent_buffer",),
            size_bytes=1,
        ),
        module.CleanupDecision(
            root=tmp_path,
            run_id=delete_dir.name,
            path=delete_dir,
            action="delete",
            reasons=(),
            size_bytes=1,
        ),
    ]

    summary = module.execute_cleanup(decisions, apply=True)

    assert keep_dir.exists()
    assert not delete_dir.exists()
    assert summary["deleted_runs"] == 1
