"""Guard against machine-specific absolute paths in tracked source files."""

from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCAN_GLOBS = ("*.py", "*.toml", "*.ps1", "*.sh", "*.ipynb", "*.yml", "*.yaml", "*.md")
SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "outputs",
    "data",
    "personal",
}
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9._-])[A-Za-z]:[\\/](?:[^\n\r\t\"'<>|]+[\\/])+[^\n\r\t\"'<>|]*"),  # Windows drive paths
    re.compile(r"/Users/[A-Za-z0-9._-]+/"),  # macOS home paths
    re.compile(r"/home/[A-Za-z0-9._-]+/"),  # Linux home paths
)


def _iter_scan_files() -> list[Path]:
    """Return tracked source files that should be scanned for absolute-path leaks."""
    files: list[Path] = []
    for pattern in SCAN_GLOBS:
        for candidate in PROJECT_ROOT.rglob(pattern):
            if any(part in SKIP_DIR_NAMES or part.startswith(".venv") for part in candidate.parts):
                continue
            files.append(candidate)
    return files


def _extract_notebook_source(path: Path) -> str:
    """Extract executable notebook source so only committed code is scanned."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    fragments: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            fragments.extend(cell.get("source", []))
    return "".join(fragments)


def _find_matches(text: str) -> list[str]:
    """Return all machine-specific absolute path matches found in one text blob."""
    matches: list[str] = []
    for pattern in ABSOLUTE_PATH_PATTERNS:
        matches.extend(pattern.findall(text))
    return matches


def test_source_files_do_not_contain_machine_specific_absolute_paths():
    """Ensure tracked sources do not hardcode developer-specific absolute paths."""
    violations: list[str] = []
    for path in _iter_scan_files():
        if path.suffix == ".ipynb":
            content = _extract_notebook_source(path)
        else:
            content = path.read_text(encoding="utf-8")
        matched = _find_matches(content)
        if matched:
            violations.append(f"{path.relative_to(PROJECT_ROOT)} -> {matched[0]}")

    assert not violations, "Hardcoded absolute paths detected:\n" + "\n".join(violations)
