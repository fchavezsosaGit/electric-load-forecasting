"""Prune superseded output run directories without breaking current evidence.

This tool exists because the repository keeps timestamped stage runs for
provenance, but long tuning cycles can easily leave hundreds of stale artifact
directories behind. Deleting runs ad hoc is risky because current docs, latest
artifacts, and control policies still reference a small subset of historical
run ids.

The cleanup policy is intentionally conservative:

- always keep stable alias folders such as ``latest/`` and stage support folders
- keep a small recent buffer per stage root so the newest developer work is not
  discarded prematurely
- keep any dated run directory whose run id is still referenced by the current
  documentation or by the current ``latest/`` artifact surface

The script can run in dry-run mode for review or apply deletions in place. It
also writes a markdown report so operators can audit what was removed and why.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d+Z$")
RUN_ID_SEARCH_PATTERN = re.compile(r"\b\d{8}T\d+Z\b")
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".ipynb"}


@dataclass(frozen=True)
class CleanupRootPolicy:
    """Describe how one artifact root should retain dated run directories."""

    root: Path
    keep_recent: int
    label: str


@dataclass(frozen=True)
class CleanupDecision:
    """Capture the keep/delete decision for one dated run directory."""

    root: Path
    run_id: str
    path: Path
    action: str
    reasons: tuple[str, ...]
    size_bytes: int


def _timestamp_utc() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _directory_size_bytes(path: Path) -> int:
    """Measure the recursive byte size for one directory."""
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def build_cleanup_policies(artifact_namespace: str) -> list[CleanupRootPolicy]:
    """Return the ordered cleanup roots for the current repository layout."""
    namespaced = {
        "stage5_namespaced": PROJECT_ROOT / "outputs" / "005_performance" / artifact_namespace,
        "stage6_namespaced": PROJECT_ROOT / "outputs" / "006_multires" / artifact_namespace,
        "stage7_namespaced": PROJECT_ROOT / "outputs" / "007_rollout" / artifact_namespace,
        "stage7_sweeps": PROJECT_ROOT
        / "outputs"
        / "007_rollout"
        / artifact_namespace
        / "challenger_sweeps",
        "stage8_namespaced": PROJECT_ROOT / "outputs" / "008_notebook_runs" / artifact_namespace,
        "stage9_namespaced": PROJECT_ROOT / "outputs" / "009_horizon_curve" / artifact_namespace,
        "stage10_namespaced": PROJECT_ROOT / "outputs" / "010_forecast_control" / artifact_namespace,
    }
    return [
        CleanupRootPolicy(PROJECT_ROOT / "outputs" / "005_performance", 3, "Stage-5 root"),
        CleanupRootPolicy(namespaced["stage5_namespaced"], 4, "Stage-5 namespaced"),
        CleanupRootPolicy(PROJECT_ROOT / "outputs" / "006_multires", 6, "Stage-6 root"),
        CleanupRootPolicy(namespaced["stage6_namespaced"], 4, "Stage-6 namespaced"),
        CleanupRootPolicy(PROJECT_ROOT / "outputs" / "007_rollout", 4, "Stage-7 root"),
        CleanupRootPolicy(namespaced["stage7_namespaced"], 12, "Stage-7 namespaced"),
        CleanupRootPolicy(namespaced["stage7_sweeps"], 8, "Stage-7 challenger sweeps"),
        CleanupRootPolicy(PROJECT_ROOT / "outputs" / "008_notebook_runs", 2, "Stage-8 notebook root"),
        CleanupRootPolicy(namespaced["stage8_namespaced"], 3, "Stage-8 notebook namespaced"),
        CleanupRootPolicy(namespaced["stage9_namespaced"], 4, "Stage-9 horizon curve"),
        CleanupRootPolicy(namespaced["stage10_namespaced"], 5, "Stage-10 forecast control"),
    ]


def build_scan_targets(artifact_namespace: str) -> list[Path]:
    """Return the doc and latest-artifact paths whose run references must survive."""
    return [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "personal" / "improvement.md",
        PROJECT_ROOT / "outputs" / "005_performance" / "latest",
        PROJECT_ROOT / "outputs" / "005_performance" / artifact_namespace / "latest",
        PROJECT_ROOT / "outputs" / "006_multires" / "latest",
        PROJECT_ROOT / "outputs" / "006_multires" / "latest_candidate",
        PROJECT_ROOT / "outputs" / "006_multires" / "latest_focus_60m",
        PROJECT_ROOT / "outputs" / "006_multires" / "latest_smoke",
        PROJECT_ROOT / "outputs" / "006_multires" / artifact_namespace / "latest",
        PROJECT_ROOT / "outputs" / "006_multires" / artifact_namespace / "latest_focus_60m",
        PROJECT_ROOT / "outputs" / "006_multires" / artifact_namespace / "latest_smoke",
        PROJECT_ROOT / "outputs" / "007_rollout" / "latest",
        PROJECT_ROOT / "outputs" / "007_rollout" / artifact_namespace / "latest",
        PROJECT_ROOT
        / "outputs"
        / "007_rollout"
        / artifact_namespace
        / "challenger_sweeps"
        / "latest",
        PROJECT_ROOT / "outputs" / "008_notebook_runs" / artifact_namespace / "latest",
        PROJECT_ROOT / "outputs" / "009_horizon_curve" / artifact_namespace / "latest",
        PROJECT_ROOT / "outputs" / "010_forecast_control" / artifact_namespace / "latest",
    ]


def _iter_text_files(target: Path) -> list[Path]:
    """Expand one file-or-directory target into the text files worth scanning."""
    if not target.exists():
        return []
    if target.is_file():
        return [target]
    return sorted(path for path in target.rglob("*") if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES)


def find_referenced_run_ids(scan_targets: list[Path]) -> set[str]:
    """Extract run ids from the documentation and latest-artifact surface."""
    run_ids: set[str] = set()
    for target in scan_targets:
        for text_file in _iter_text_files(target):
            try:
                text = text_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            run_ids.update(RUN_ID_SEARCH_PATTERN.findall(text))
    return run_ids


def plan_cleanup(
    policies: list[CleanupRootPolicy],
    referenced_run_ids: set[str],
) -> list[CleanupDecision]:
    """Plan keep/delete decisions for all configured cleanup roots."""
    decisions: list[CleanupDecision] = []
    for policy in policies:
        if not policy.root.exists():
            continue
        dated_dirs = sorted(
            (
                child
                for child in policy.root.iterdir()
                if child.is_dir() and RUN_ID_PATTERN.match(child.name)
            ),
            key=lambda child: child.name,
            reverse=True,
        )
        keep_recent_ids = {child.name for child in dated_dirs[: policy.keep_recent]}
        for child in dated_dirs:
            reasons: list[str] = []
            if child.name in keep_recent_ids:
                reasons.append("recent_buffer")
            if child.name in referenced_run_ids:
                reasons.append("referenced_by_docs_or_latest")
            action = "keep" if reasons else "delete"
            decisions.append(
                CleanupDecision(
                    root=policy.root,
                    run_id=child.name,
                    path=child,
                    action=action,
                    reasons=tuple(reasons),
                    size_bytes=_directory_size_bytes(child),
                )
            )
    return decisions


def execute_cleanup(decisions: list[CleanupDecision], apply: bool) -> dict[str, int]:
    """Delete planned directories when apply mode is enabled and return counts."""
    deleted_runs = 0
    deleted_bytes = 0
    for decision in decisions:
        if decision.action != "delete":
            continue
        if apply and decision.path.exists():
            shutil.rmtree(decision.path)
        deleted_runs += 1
        deleted_bytes += decision.size_bytes
    return {"deleted_runs": deleted_runs, "deleted_bytes": deleted_bytes}


def render_report(
    decisions: list[CleanupDecision],
    referenced_run_ids: set[str],
    *,
    apply: bool,
    artifact_namespace: str,
) -> str:
    """Render a markdown summary of the cleanup decision surface."""
    kept = [decision for decision in decisions if decision.action == "keep"]
    deleted = [decision for decision in decisions if decision.action == "delete"]
    lines = [
        "# Output Cleanup Report",
        "",
        f"- Generated at: `{_timestamp_utc()}`",
        f"- Artifact namespace: `{artifact_namespace}`",
        f"- Mode: `{'apply' if apply else 'dry_run'}`",
        f"- Referenced run ids discovered: `{len(referenced_run_ids)}`",
        f"- Kept dated runs: `{len(kept)}`",
        f"- Deleted dated runs: `{len(deleted)}`",
        f"- Estimated deleted size: `{sum(item.size_bytes for item in deleted) / 1_048_576:.2f} MB`",
        "",
        "## Policy",
        "",
        "- Keep any run id still referenced by docs or the current `latest/` artifact surface.",
        "- Keep a small recent buffer per stage root so active work is not lost.",
        "- Delete only dated run directories that are neither referenced nor recent.",
        "",
        "## Deleted Runs",
        "",
    ]
    if deleted:
        for decision in sorted(deleted, key=lambda item: str(item.path)):
            lines.append(
                f"- `{decision.path.relative_to(PROJECT_ROOT)}` "
                f"({decision.size_bytes / 1_048_576:.2f} MB)"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Kept Runs With Reasons", ""])
    for decision in sorted(kept, key=lambda item: str(item.path)):
        reason_text = ", ".join(decision.reasons)
        lines.append(
            f"- `{decision.path.relative_to(PROJECT_ROOT)}` kept because `{reason_text}` "
            f"({decision.size_bytes / 1_048_576:.2f} MB)"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the output cleanup tool."""
    parser = argparse.ArgumentParser(description="Prune superseded output runs while preserving current evidence.")
    parser.add_argument(
        "--artifact-namespace",
        default="commercial_facility",
        help="Artifact namespace whose namespaced output roots should be cleaned.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete directories instead of only printing the planned cleanup.",
    )
    parser.add_argument(
        "--report-path",
        default="personal/output_cleanup_report.md",
        help="Markdown path where the cleanup report should be written.",
    )
    parser.add_argument(
        "--json-report-path",
        default="personal/output_cleanup_report.json",
        help="JSON path where the structured cleanup report should be written.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the cleanup planner, optionally delete directories, and write reports."""
    args = parse_args()
    policies = build_cleanup_policies(args.artifact_namespace)
    scan_targets = build_scan_targets(args.artifact_namespace)
    referenced_run_ids = find_referenced_run_ids(scan_targets)
    decisions = plan_cleanup(policies, referenced_run_ids)
    execution_summary = execute_cleanup(decisions, apply=bool(args.apply))

    report_path = PROJECT_ROOT / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = render_report(
        decisions,
        referenced_run_ids,
        apply=bool(args.apply),
        artifact_namespace=args.artifact_namespace,
    )
    report_path.write_text(report_text, encoding="utf-8")

    json_report_path = PROJECT_ROOT / args.json_report_path
    json_report_path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "generated_at_utc": _timestamp_utc(),
        "artifact_namespace": args.artifact_namespace,
        "mode": "apply" if args.apply else "dry_run",
        "referenced_run_ids": sorted(referenced_run_ids),
        "execution_summary": execution_summary,
        "decisions": [
            {
                "root": str(decision.root.relative_to(PROJECT_ROOT)),
                "run_id": decision.run_id,
                "path": str(decision.path.relative_to(PROJECT_ROOT)),
                "action": decision.action,
                "reasons": list(decision.reasons),
                "size_bytes": decision.size_bytes,
            }
            for decision in decisions
        ],
    }
    json_report_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    print(report_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
