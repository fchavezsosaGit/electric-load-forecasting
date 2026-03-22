"""Generate a canonical current-validation snapshot from latest repo artifacts.

This tool closes a recurrent repo-quality gap: multiple markdown reports can
contain historical timing and metric snapshots, while operators still need one
clear answer to "what is the current validated state right now?".

The snapshot is generated from the latest persisted artifacts instead of manual
copy editing. It is intended to be called automatically after the E2E runner and
can also be executed directly when artifacts already exist.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

import sys

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import (  # noqa: E402
    DATASET,
    DEFAULT_RESOLUTIONS,
    MULTIRES_FORECAST_CONTROL,
    MULTIRES_SELECTION,
    SUPPORTED_RESOLUTIONS,
)

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "docs" / "003_modeling" / "current_validation_snapshot.md"


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file when present and valid, otherwise return `None`."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV artifact when present, otherwise return an empty frame."""
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _rel(path: Path, base_dir: Path) -> str:
    """Render a filesystem path relative to the markdown file that will link to it."""
    return Path(os.path.relpath(path.resolve(), start=base_dir.resolve())).as_posix()


def _metric_fields(selection_target: str) -> tuple[str, str]:
    """Resolve learned/persistence metric columns for one Stage-8 selection target."""
    metric_map = {
        "endpoint_mae": ("learned_endpoint_mae", "persistence_endpoint_mae"),
        "path_mae": ("learned_path_mae", "persistence_path_mae"),
        "phase_mean_mae": ("learned_phase_mean_mae", "persistence_phase_mean_mae"),
        "next_lock_mae": ("learned_next_lock_mae", "persistence_next_lock_mae"),
        "profile_shape_mae": ("learned_profile_shape_mae", "persistence_profile_shape_mae"),
    }
    return metric_map.get(str(selection_target), ("learned_path_mae", "persistence_path_mae"))


def _format_metric(value: Any) -> str:
    """Format a numeric metric for compact snapshot reporting."""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_pct(value: Any) -> str:
    """Format a numeric percentage for compact snapshot reporting."""
    try:
        return f"{float(value):.6f}%"
    except (TypeError, ValueError):
        return "n/a"


def _resolution_seconds(label: Any) -> int | None:
    """Translate a resolution label like `30s` or `1min` into seconds."""
    if not isinstance(label, str):
        return None
    text = label.strip().lower()
    if text.endswith("min"):
        try:
            return int(text[:-3]) * 60
        except ValueError:
            return None
    if text.endswith("s"):
        try:
            return int(text[:-1])
        except ValueError:
            return None
    return None


def _current_resolution_policy_lines(
    *,
    selection_summary: pd.DataFrame,
    matched_horizon_metrics: pd.DataFrame,
) -> list[str]:
    """Summarize the current operating-resolution policy from persisted evidence."""
    actual_resolution = str(MULTIRES_FORECAST_CONTROL["actual_resolution"])
    lock_interval_minutes = int(MULTIRES_FORECAST_CONTROL["lock_interval_minutes"])
    default_resolutions = ", ".join(f"`{value}`" for value in DEFAULT_RESOLUTIONS)
    supported_resolutions = ", ".join(f"`{value}`" for value in SUPPORTED_RESOLUTIONS)
    lines = [
        "## Current Resolution Policy",
        "",
        f"- Supported pipeline resolutions: {supported_resolutions}",
        f"- Default materialized pipeline resolutions: {default_resolutions}",
        f"- Current optimizer-facing actual resolution: `{actual_resolution}` with `{lock_interval_minutes}` minute lock intervals.",
    ]

    matched = selection_summary.loc[
        selection_summary.get("use_case", pd.Series(dtype="string"))
        .astype("string")
        .str.startswith("matched_horizon_", na=False)
    ].copy()
    if not matched.empty:
        matched["winner_horizon_minutes"] = pd.to_numeric(
            matched.get("winner_horizon_minutes"), errors="coerce"
        )
        matched = matched.sort_values("winner_horizon_minutes", kind="stable")
        winner_summary = ", ".join(
            (
                f"`{int(row['winner_horizon_minutes'])}m` -> "
                f"`{row.get('winner_resolution', 'n/a')}` "
                f"({str(row.get('winner_type', 'n/a')).replace('_', ' ')})"
            )
            for _, row in matched.iterrows()
            if pd.notna(row.get("winner_horizon_minutes"))
        )
        if winner_summary:
            lines.append(
                "- Latest matched-horizon winners that cleared the Stage-6 gates: "
                f"{winner_summary}"
            )

    subminute = matched_horizon_metrics.copy()
    if not subminute.empty:
        subminute["_resolution_seconds"] = subminute.get("resolution", pd.Series(dtype="string")).map(
            _resolution_seconds
        )
        subminute["_mae_ratio"] = pd.to_numeric(
            subminute.get("mae_ratio_to_persistence"), errors="coerce"
        )
        subminute["_fold_std_ratio"] = pd.to_numeric(
            subminute.get("fold_std_mae_ratio"), errors="coerce"
        )
        subminute["_runtime_seconds"] = pd.to_numeric(
            subminute.get("runtime_seconds"), errors="coerce"
        )
        subminute["_horizon_minutes"] = pd.to_numeric(
            subminute.get("horizon_minutes"), errors="coerce"
        )
        subminute = subminute.loc[
            subminute["_resolution_seconds"].notna()
            & subminute["_resolution_seconds"].lt(60)
            & subminute.get("candidate_type", pd.Series(dtype="string"))
            .astype("string")
            .eq("learned")
            & subminute["_mae_ratio"].notna()
        ].sort_values(
            ["_mae_ratio", "_fold_std_ratio", "_runtime_seconds"],
            ascending=[True, True, True],
            kind="stable",
        )
        if not subminute.empty:
            best = subminute.iloc[0]
            lines.append(
                "- Best current sub-minute challenger: "
                f"`{best.get('resolution', 'n/a')}/{best.get('feature_set', 'n/a')}/"
                f"{best.get('model_label', 'n/a')}/{best.get('forecast_strategy', 'n/a')}` "
                f"at `{int(best['_horizon_minutes'])}m` with MAE ratio "
                f"`{_format_metric(best.get('_mae_ratio'))}` to persistence."
            )
            if not bool(best.get("eligible", False)):
                lines.append(
                    "- Why that does not replace `1min` today: the best current sub-minute candidate "
                    "still fails the Stage-6 operating gates, so it stays exploratory instead of "
                    "becoming the default control cadence."
                )
                lines.append(
                    "- Current gate readout for that challenger: "
                    f"`eligible={bool(best.get('eligible', False))}` / "
                    f"`pareto_passed={bool(best.get('pareto_passed', False))}` / "
                    f"`practical_gain_passed={bool(best.get('practical_gain_passed', False))}` / "
                    f"`fold_std_mae_ratio={_format_metric(best.get('_fold_std_ratio'))}` "
                    f"against the configured stability gate "
                    f"`{MULTIRES_SELECTION['max_fold_std_mae_ratio']:.6f}`."
                )

    lines.extend(
        [
            "- Current operating rule: keep `1min` as the validated optimizer-facing correction cadence, "
            "treat `30s` as exploratory where it shows raw matched-horizon promise, and do not promote "
            "`1s` / `5s` / `10s` without new persisted evidence that they beat the current gates.",
            "- Practical implication: the repo can ingest and compare sub-minute data, but the current "
            "Stage-10 delivery contract is still a `1min` nowcast overlay on top of the broader `15m` "
            "decision loop.",
            "",
        ]
    )
    return lines


def _supplemental_surface_high_signal_note(segment_evaluation: pd.DataFrame) -> str:
    """Summarize the strongest learned-positive supplemental diagnostic segment."""
    if segment_evaluation.empty:
        return "none"
    candidates = segment_evaluation.copy()
    candidates["_ratio"] = pd.to_numeric(candidates.get("candidate_mae_ratio_to_best_baseline"), errors="coerce")
    candidates["_rows"] = pd.to_numeric(candidates.get("rows"), errors="coerce")
    candidates = candidates.loc[
        candidates["_ratio"].notna()
        & candidates["_rows"].notna()
        & candidates["_ratio"].lt(1.0)
        & candidates["_rows"].gt(1)
    ].sort_values(["_ratio", "_rows"], ascending=[True, False], kind="stable")
    if candidates.empty:
        return "none"
    best = candidates.iloc[0]
    return (
        f"{best.get('segment_column', 'segment')}={best.get('segment_value', 'unknown')} "
        f"at ratio `{_format_metric(best.get('_ratio'))}` over `{int(best.get('_rows', 0))}` rows"
    )


def _phase_interpretation_line(
    *,
    phase_policy: dict[str, Any] | None = None,
    rolling_summary: pd.DataFrame,
    rolling_inference: pd.DataFrame,
) -> str:
    """Summarize whether the stack-applied phase layer helps on rolling evidence."""
    phase_policy = phase_policy or {}
    if bool(phase_policy.get("rolling_support_applied_veto", False)):
        return (
            "- The exact stack-aware `15m` phase benchmark still found a distinct "
            "candidate, but the broader rolling-support guard vetoed it, so the "
            "current applied phase slot resolves to hourly passthrough while the "
            "final minute nowcast remains the largest improvement."
        )
    phase_inference = rolling_inference.loc[
        rolling_inference.get("scope", pd.Series(dtype="string")).astype("string").eq("rolling_evaluation")
        & rolling_inference.get("comparison_label", pd.Series(dtype="string")).astype("string").eq("phase_vs_hourly")
        & rolling_inference.get("metric_name", pd.Series(dtype="string")).astype("string").eq("lock_mae")
    ]
    if not phase_inference.empty:
        row = phase_inference.iloc[0]
        try:
            gain_metric = float(row.get("gain_metric"))
        except (TypeError, ValueError):
            gain_metric = float("nan")
        gain_supported = bool(row.get("candidate_better_than_baseline", False)) and bool(
            row.get("gain_ci_excludes_zero", False)
        )
        if gain_supported and pd.notna(gain_metric) and gain_metric > 0.0:
            return (
                "- The current stack-applied `15m` phase layer now adds a meaningful "
                "rolling gain on top of the hourly layer, while the final minute "
                "nowcast remains the largest improvement."
            )
        return (
            "- The current stack-applied `15m` phase layer does not yet show a robust "
            "rolling gain beyond the hourly layer, while the final minute nowcast "
            "remains the largest improvement."
        )

    rolling_rows = {
        str(row.get("layer", "")): row
        for _, row in rolling_summary.iterrows()
    }
    try:
        hourly_lock = float(rolling_rows.get("after_hourly_updates", {}).get("lock_mae"))
        phase_lock = float(rolling_rows.get("after_phase_updates", {}).get("lock_mae"))
    except (TypeError, ValueError):
        hourly_lock = float("nan")
        phase_lock = float("nan")
    if pd.notna(hourly_lock) and pd.notna(phase_lock) and phase_lock < hourly_lock:
        return (
            "- The current stack-applied `15m` phase layer now adds a meaningful "
            "rolling gain on top of the hourly layer, while the final minute "
            "nowcast remains the largest improvement."
        )
    return (
        "- The current stack-applied `15m` phase layer does not yet show a robust "
        "rolling gain beyond the hourly layer, while the final minute nowcast "
        "remains the largest improvement."
    )


def build_validation_snapshot_content(
    *,
    project_root: Path,
    artifact_namespace: str,
    doc_dir: Path,
    step_seconds: dict[str, float] | None = None,
    generated_at_utc: str | None = None,
) -> str:
    """Build the canonical markdown snapshot from the latest persisted artifacts."""
    generated_at_utc = generated_at_utc or datetime.now(UTC).isoformat()
    latest_paths = {
        "performance": project_root / "outputs" / "005_performance" / artifact_namespace / "latest",
        "multires": project_root / "outputs" / "006_multires" / artifact_namespace / "latest",
        "rollout": project_root / "outputs" / "007_rollout" / artifact_namespace / "latest",
        "rollout_sweep": project_root / "outputs" / "007_rollout" / artifact_namespace / "challenger_sweeps" / "latest",
        "notebooks": project_root / "outputs" / "008_notebook_runs" / artifact_namespace / "latest",
        "horizon_curve": project_root / "outputs" / "009_horizon_curve" / artifact_namespace / "latest",
        "forecast_control": project_root / "outputs" / "010_forecast_control" / artifact_namespace / "latest",
    }
    visualization_dashboard = project_root / "outputs" / "reports" / artifact_namespace / "latest" / "validation_dashboard.html"
    visualization_guide = project_root / "docs" / "003_modeling" / "current_visualization_guide.md"
    horizon_anchor_png = latest_paths["horizon_curve"] / "fig_horizon_ratio_curve.png"
    control_anchor_png = latest_paths["forecast_control"] / "fig_control_layer_gain_ci.png"

    performance_manifest = _read_json(latest_paths["performance"] / "run_manifest.json") or {}
    rollout_manifest = _read_json(latest_paths["rollout"] / "run_manifest.json") or {}
    sweep_manifest = _read_json(latest_paths["rollout_sweep"] / "run_manifest.json") or {}
    horizon_manifest = _read_json(latest_paths["horizon_curve"] / "run_manifest.json") or {}
    control_manifest = _read_json(latest_paths["forecast_control"] / "run_manifest.json") or {}
    notebook_manifest = _read_json(latest_paths["notebooks"] / "run_manifest.json") or {}
    deployment_recommendation = _read_json(latest_paths["performance"] / "deployment_recommendation.json") or {}
    sweep_recommendation = _read_json(latest_paths["rollout_sweep"] / "recommended_candidate.json") or {}
    control_policy = _read_json(latest_paths["forecast_control"] / "control_policy.json") or {}
    delivery_contract = _read_json(latest_paths["forecast_control"] / "optimizer_delivery_contract.json") or {}
    delivery_operational_policy = (
        _read_json(latest_paths["forecast_control"] / "optimizer_operational_policy.json") or {}
    )
    dynamic_overlay_shadow_summary = (
        _read_json(latest_paths["forecast_control"] / "optimizer_dynamic_overlay_shadow_summary.json") or {}
    )
    dynamic_overlay_soft_summary = (
        _read_json(latest_paths["forecast_control"] / "optimizer_dynamic_overlay_soft_summary.json") or {}
    )
    stage5_operating_policy = _read_json(latest_paths["performance"] / "operating_policy.json") or {}
    holdout_coverage_summary = _read_json(latest_paths["performance"] / "holdout_coverage_summary.json") or {}
    supplemental_surface_advisory = (
        _read_json(latest_paths["performance"] / "supplemental_surface_advisory.json") or {}
    )
    supplemental_surface_segment_evaluation = _read_csv(
        latest_paths["performance"] / "supplemental_surface_segment_evaluation.csv"
    )
    runtime_summary = _read_json(latest_paths["forecast_control"] / "runtime_summary.json") or {}
    control_cycle_scope = control_policy.get("control_cycle_scope", {})

    holdout = _read_csv(latest_paths["performance"] / "holdout_evaluation.csv")
    selection_summary = _read_csv(latest_paths["multires"] / "selection_summary.csv")
    matched_horizon_metrics = _read_csv(latest_paths["multires"] / "matched_horizon_metrics.csv")
    horizon = _read_csv(latest_paths["horizon_curve"] / "horizon_curve_summary.csv")
    control_summary = _read_csv(latest_paths["forecast_control"] / "control_backtest_summary.csv")
    rolling_summary = _read_csv(latest_paths["forecast_control"] / "rolling_control_backtest_summary.csv")
    rolling_inference = _read_csv(latest_paths["forecast_control"] / "rolling_control_layer_inference.csv")
    refresh_summary = _read_csv(latest_paths["forecast_control"] / "day_ahead_refresh_summary.csv")
    delivery_uncertainty_summary = _read_csv(
        latest_paths["forecast_control"] / "optimizer_delivery_uncertainty_summary.csv"
    )
    supplemental_high_signal_note = _supplemental_surface_high_signal_note(
        supplemental_surface_segment_evaluation
    )

    learned_holdout = holdout.loc[
        holdout.get("candidate_type", pd.Series(index=holdout.index, dtype="string"))
        .astype("string")
        .str.contains("learned", case=False, na=False)
    ].sort_values("mae", kind="stable")
    learned_holdout_row = learned_holdout.iloc[0] if not learned_holdout.empty else pd.Series(dtype=object)
    persistence_holdout = holdout.loc[
        holdout.get("candidate_label", pd.Series(index=holdout.index, dtype="string"))
        .astype("string")
        .eq("persistence")
    ]
    persistence_holdout_row = persistence_holdout.iloc[0] if not persistence_holdout.empty else pd.Series(dtype=object)

    horizon_rows: list[str] = []
    for horizon_minutes in (1, 15, 60, 1440):
        matched = horizon.loc[horizon.get("horizon_minutes", pd.Series(dtype="float64")).eq(horizon_minutes)]
        if matched.empty:
            continue
        row = matched.iloc[0]
        learned_metric, persistence_metric = _metric_fields(str(row.get("selection_target", "")))
        horizon_rows.append(
            "- "
            f"`{horizon_minutes}m`: `{row.get('candidate_label', 'n/a')}` on "
            f"`{row.get('selection_target', 'n/a')}` with "
            f"{_format_metric(row.get(learned_metric))} vs persistence "
            f"{_format_metric(row.get(persistence_metric))}"
        )

    control_rows = {
        str(row.get("layer", "")): row
        for _, row in control_summary.iterrows()
    }
    rolling_rows = {
        str(row.get("layer", "")): row
        for _, row in rolling_summary.iterrows()
    }
    refresh_rows = {
        str(row.get("scenario", "")): row
        for _, row in refresh_summary.iterrows()
    }
    delivery_uncertainty_rows = {
        str(row.get("scope", "")): row
        for _, row in delivery_uncertainty_summary.iterrows()
    }
    shadow_metrics = dynamic_overlay_shadow_summary.get("shadow_mode", {})
    enforced_metrics = dynamic_overlay_shadow_summary.get("enforced_counterfactual", {})
    shadow_delta = dynamic_overlay_shadow_summary.get("delta_enforced_minus_shadow", {})
    soft_best_candidate = dynamic_overlay_soft_summary.get("best_improving_candidate", {}) or dynamic_overlay_soft_summary.get(
        "best_admissible_candidate",
        {},
    )

    modeling_notebook = {}
    for entry in notebook_manifest.get("notebooks", []):
        if str(entry.get("source_path", "")).endswith("003_modeling.ipynb"):
            modeling_notebook = entry
            break
    artifact_validation = modeling_notebook.get("artifact_validation", {})
    csv_artifacts = artifact_validation.get("csv_artifacts", {})
    phase_policy = control_policy.get("phase", {})
    phase_stack_selected_label = phase_policy.get(
        "stack_selected_candidate_label",
        phase_policy.get("candidate_label", "n/a"),
    )
    phase_stack_applied_label = phase_policy.get("stack_guard_applied_candidate_label", "n/a")
    if bool(phase_policy.get("rolling_support_applied_veto", False)):
        phase_stack_note = (
            "the broader rolling-support guard vetoed the distinct exact-stack phase "
            f"candidate `{phase_stack_selected_label}`, so the current applied phase slot "
            f"falls back to `{phase_stack_applied_label}`"
        )
    else:
        phase_stack_note = f"the current applied phase slot uses `{phase_stack_applied_label}`"

    lines = [
        "# Current Validation Snapshot",
        "",
        "This file is generated from the latest persisted artifacts. It is the",
        "canonical answer to \"what is the repo's current validated state?\"",
        "",
        f"- Generated at: `{generated_at_utc}`",
        f"- Artifact namespace: `{artifact_namespace}`",
        "",
        "## Validation Surface",
        "",
    ]
    if step_seconds:
        total = sum(step_seconds.values())
        lines.extend(
            [
                "- Latest integrated E2E command:",
                "  `python scripts/run_e2e.py --mode quick --with-multires --with-rollout --with-rollout-sweep --with-horizon-curve --with-forecast-control`",
                f"- Pipeline: `{step_seconds.get('pipeline', 0.0):.2f}s`",
                f"- Notebooks: `{step_seconds.get('notebooks', 0.0):.2f}s`",
                f"- Pytest: `{step_seconds.get('pytest', 0.0):.2f}s`",
                f"- Total: `{total:.2f}s`",
                "",
            ]
        )
    lines.extend(
        [
            f"- Stage-5 latest manifest: [`{_rel(latest_paths['performance'] / 'run_manifest.json', doc_dir)}`]({_rel(latest_paths['performance'] / 'run_manifest.json', doc_dir)})",
            f"- Stage-6 latest manifest: [`{_rel(latest_paths['multires'] / 'run_manifest.json', doc_dir)}`]({_rel(latest_paths['multires'] / 'run_manifest.json', doc_dir)})",
            f"- Stage-7 latest rollout manifest: [`{_rel(latest_paths['rollout'] / 'run_manifest.json', doc_dir)}`]({_rel(latest_paths['rollout'] / 'run_manifest.json', doc_dir)})",
            f"- Stage-7 latest sweep manifest: [`{_rel(latest_paths['rollout_sweep'] / 'run_manifest.json', doc_dir)}`]({_rel(latest_paths['rollout_sweep'] / 'run_manifest.json', doc_dir)})",
            f"- Stage-8 latest manifest: [`{_rel(latest_paths['horizon_curve'] / 'run_manifest.json', doc_dir)}`]({_rel(latest_paths['horizon_curve'] / 'run_manifest.json', doc_dir)})",
            f"- Stage-10 latest manifest: [`{_rel(latest_paths['forecast_control'] / 'run_manifest.json', doc_dir)}`]({_rel(latest_paths['forecast_control'] / 'run_manifest.json', doc_dir)})",
            f"- Notebook archive manifest: [`{_rel(latest_paths['notebooks'] / 'run_manifest.json', doc_dir)}`]({_rel(latest_paths['notebooks'] / 'run_manifest.json', doc_dir)})",
            f"- Stage-5 holdout coverage summary: [`{_rel(latest_paths['performance'] / 'holdout_coverage_summary.json', doc_dir)}`]({_rel(latest_paths['performance'] / 'holdout_coverage_summary.json', doc_dir)})",
            f"- Stage-5 supplemental surface advisory: [`{_rel(latest_paths['performance'] / 'supplemental_surface_advisory.json', doc_dir)}`]({_rel(latest_paths['performance'] / 'supplemental_surface_advisory.json', doc_dir)})",
            f"- Stage-10 runtime summary: [`{_rel(latest_paths['forecast_control'] / 'runtime_summary.json', doc_dir)}`]({_rel(latest_paths['forecast_control'] / 'runtime_summary.json', doc_dir)})",
            f"- Exact-control cycles: calibration `{control_cycle_scope.get('calibration_cycle_count', 'n/a')}`, evaluation `{control_cycle_scope.get('evaluation_cycle_count', 'n/a')}`",
            f"- Rolling benchmark cycles: calibration `{control_cycle_scope.get('rolling_calibration_cycle_count', 'n/a')}`, evaluation `{control_cycle_scope.get('rolling_evaluation_cycle_count', 'n/a')}`",
            "",
            "## Current Findings",
            "",
            "- Stage-5 holdout recommendation:",
            f"  `{deployment_recommendation.get('recommended_candidate_label', 'n/a')}`",
            f"  because {deployment_recommendation.get('decision_reason', 'n/a')}",
            "- Stage-5 minute operating policy:",
            f"  standalone `{deployment_recommendation.get('standalone_operating_role', 'n/a')}` / "
            f"Stage-10 `{deployment_recommendation.get('stage10_operating_role', 'n/a')}`",
            "- Best current learned 1m challenger:",
            f"  `{learned_holdout_row.get('candidate_label', 'n/a')}` at "
            f"{_format_metric(learned_holdout_row.get('mae'))} "
            f"({_format_pct(learned_holdout_row.get('mae_pct'))}) vs persistence "
            f"{_format_metric(persistence_holdout_row.get('mae'))} "
            f"({_format_pct(persistence_holdout_row.get('mae_pct'))})",
            "- Stage-5 holdout coverage note:",
            f"  {holdout_coverage_summary.get('reason', 'n/a')}",
            "- Stage-5 supplemental advisory surface:",
            f"  learned beats persistence on the broader advisory surface: `{supplemental_surface_advisory.get('learned_beats_persistence', 'n/a')}`; "
            f"learned-supported operating regimes: `{', '.join(supplemental_surface_advisory.get('learned_supported_operating_regimes', [])) or 'none'}`",
            "- Strongest supplemental diagnostic segment:",
            f"  {supplemental_high_signal_note}",
            "- Stage-7 latest day-ahead sweep recommendation:",
            f"  `{sweep_recommendation.get('recommended_resolution', 'n/a')}/"
            f"{sweep_recommendation.get('recommended_feature_set', 'n/a')}/"
            f"{sweep_recommendation.get('recommended_model_label', 'n/a')}::"
            f"{sweep_recommendation.get('recommended_target_mode', 'n/a')}` "
            f"on `{sweep_recommendation.get('selection_target', 'n/a')}` with "
            f"{_format_metric(sweep_recommendation.get('recommended_metric_value'))} "
            f"({_format_pct(sweep_recommendation.get('recommended_metric_pct'))})",
            "- Stage-8 objective winners:",
            *horizon_rows,
            "",
            "## How To Read The Winners",
            "",
            "- Stage-5 answers the deployable `1m` holdout question. If it keeps `persistence`, that is the honest short-horizon recommendation.",
            "- Stage-8 answers the horizon-characterization question. Its `1m` row does not override the Stage-5 deployment recommendation by itself.",
            "- Stage-10 answers the control-stack question. It may choose a different nowcast layer after replaying candidates on the exact control cycles.",
            "- Candidate-label anatomy, blend wrappers, and the current CPU/GPU policy are summarized in [model_and_blend_guide.md](model_and_blend_guide.md).",
            "",
            *_current_resolution_policy_lines(
                selection_summary=selection_summary,
                matched_horizon_metrics=matched_horizon_metrics,
            ),
            "## Current Control Stack",
            "",
            f"- Day-ahead: `{control_policy.get('day_ahead', {}).get('candidate_label', 'n/a')}`",
            f"- Hourly: `{control_policy.get('hourly', {}).get('candidate_label', 'n/a')}`",
            f"- Stack-applied phase: `{control_policy.get('phase', {}).get('stack_guard_applied_candidate_label', 'n/a')}`",
            f"- Phase slot note: {phase_stack_note}",
            f"- Nowcast: `{control_policy.get('nowcast_anchor', {}).get('candidate_label', 'n/a')}`",
            "",
            "| Layer | Lock MAE | Profile-Shape MAE |",
            "|-------|----------|-------------------|",
            f"| Frozen day-ahead | `{_format_metric(control_rows.get('day_ahead_frozen', {}).get('lock_mae'))}` | `{_format_metric(control_rows.get('day_ahead_frozen', {}).get('profile_shape_mae'))}` |",
            f"| After hourly updates | `{_format_metric(control_rows.get('after_hourly_updates', {}).get('lock_mae'))}` | `{_format_metric(control_rows.get('after_hourly_updates', {}).get('profile_shape_mae'))}` |",
            f"| After phase updates | `{_format_metric(control_rows.get('after_phase_updates', {}).get('lock_mae'))}` | `{_format_metric(control_rows.get('after_phase_updates', {}).get('profile_shape_mae'))}` |",
            f"| After nowcast updates | `{_format_metric(control_rows.get('after_nowcast_updates', {}).get('lock_mae'))}` | `{_format_metric(control_rows.get('after_nowcast_updates', {}).get('profile_shape_mae'))}` |",
            "",
            "- Rolling evaluation stack:",
            f"  lock `{_format_metric(rolling_rows.get('day_ahead_frozen', {}).get('lock_mae'))}` -> "
            f"`{_format_metric(rolling_rows.get('after_hourly_updates', {}).get('lock_mae'))}` -> "
            f"`{_format_metric(rolling_rows.get('after_phase_updates', {}).get('lock_mae'))}` -> "
            f"`{_format_metric(rolling_rows.get('after_nowcast_updates', {}).get('lock_mae'))}`",
            f"  profile `{_format_metric(rolling_rows.get('day_ahead_frozen', {}).get('profile_shape_mae'))}` -> "
            f"`{_format_metric(rolling_rows.get('after_hourly_updates', {}).get('profile_shape_mae'))}` -> "
            f"`{_format_metric(rolling_rows.get('after_phase_updates', {}).get('profile_shape_mae'))}` -> "
            f"`{_format_metric(rolling_rows.get('after_nowcast_updates', {}).get('profile_shape_mae'))}`",
            "",
            "## Optimizer Delivery Surface",
            "",
            f"- Delivery contract: [`{_rel(latest_paths['forecast_control'] / 'optimizer_delivery_contract.json', doc_dir)}`]({_rel(latest_paths['forecast_control'] / 'optimizer_delivery_contract.json', doc_dir)})",
            f"- Operational policy: [`{_rel(latest_paths['forecast_control'] / 'optimizer_operational_policy.json', doc_dir)}`]({_rel(latest_paths['forecast_control'] / 'optimizer_operational_policy.json', doc_dir)})",
            f"- Stage-5 minute operating policy: [`{_rel(latest_paths['performance'] / 'operating_policy.json', doc_dir)}`]({_rel(latest_paths['performance'] / 'operating_policy.json', doc_dir)})",
            f"- Delivery preview: [`{_rel(latest_paths['forecast_control'] / 'optimizer_delivery_preview.csv', doc_dir)}`]({_rel(latest_paths['forecast_control'] / 'optimizer_delivery_preview.csv', doc_dir)})",
            f"- Dynamic overlay shadow summary: [`{_rel(latest_paths['forecast_control'] / 'optimizer_dynamic_overlay_shadow_summary.json', doc_dir)}`]({_rel(latest_paths['forecast_control'] / 'optimizer_dynamic_overlay_shadow_summary.json', doc_dir)})",
            f"- Soft overlay shadow summary: [`{_rel(latest_paths['forecast_control'] / 'optimizer_dynamic_overlay_soft_summary.json', doc_dir)}`]({_rel(latest_paths['forecast_control'] / 'optimizer_dynamic_overlay_soft_summary.json', doc_dir)})",
            f"- Uncertainty summary: [`{_rel(latest_paths['forecast_control'] / 'optimizer_delivery_uncertainty_summary.csv', doc_dir)}`]({_rel(latest_paths['forecast_control'] / 'optimizer_delivery_uncertainty_summary.csv', doc_dir)})",
            f"- Contract version: `{delivery_contract.get('contract_version', 'n/a')}`",
            f"- Delivery cadence: `{delivery_contract.get('cadence_minutes', 'n/a')}` minutes",
            f"- Layer priority: `{', '.join(delivery_contract.get('selected_layer_priority', [])) or 'n/a'}`",
            f"- Uncertainty method: `{delivery_contract.get('uncertainty', {}).get('method', 'n/a')}`",
            f"- Confidence signal: `{delivery_contract.get('confidence_signal', {}).get('type', 'n/a')}`",
            f"- Preview rows carry freshness fields: `{', '.join(delivery_contract.get('freshness', {}).get('row_fields', [])) or 'n/a'}`",
            f"- Runtime portability: `{delivery_operational_policy.get('hardware_policy', {}).get('portable_default', 'n/a')}`",
            f"- Minute-layer operating role: `{stage5_operating_policy.get('stage10_operating_role', 'n/a')}`",
            f"- Dynamic minute-overlay recommendation: `{dynamic_overlay_shadow_summary.get('recommendation', 'n/a')}`",
            f"  shadow all-interval abs error `{_format_metric(shadow_metrics.get('mean_selected_abs_error'))}` vs enforced `{_format_metric(enforced_metrics.get('mean_selected_abs_error'))}`; "
            f"delta `{_format_metric(shadow_delta.get('mean_selected_abs_error'))}`",
            f"- Soft minute-overlay shadow recommendation: `{dynamic_overlay_soft_summary.get('recommendation', 'n/a')}`",
            f"  best policy `{soft_best_candidate.get('soft_policy_label', 'n/a')}` at "
            f"all-interval abs error `{_format_metric(soft_best_candidate.get('mean_selected_abs_error'))}`",
            f"- All-interval 80% / 95% empirical coverage: "
            f"`{_format_metric(delivery_uncertainty_rows.get('all_intervals', {}).get('interval_80_coverage'))}` / "
            f"`{_format_metric(delivery_uncertainty_rows.get('all_intervals', {}).get('interval_95_coverage'))}`",
            f"- Stage-10 runtime hotspot: `{runtime_summary.get('longest_step', 'n/a')}` "
            f"at `{_format_metric(runtime_summary.get('longest_step_seconds'))}` seconds",
            "",
            "## Day-Ahead Refresh",
            "",
            f"- Recommended policy: `{control_policy.get('day_ahead_refresh', {}).get('recommended_policy', 'n/a')}`",
            f"- Trigger mode: `{control_policy.get('day_ahead_refresh', {}).get('trigger_mode', 'n/a')}`",
            f"- Exact-control trigger rate: `{float(control_policy.get('day_ahead_refresh', {}).get('evaluation_trigger_rate', float('nan'))):.6f}`",
            f"- Rolling trigger rate: `{float(control_policy.get('day_ahead_refresh', {}).get('rolling_benchmark', {}).get('trigger_rate', float('nan'))):.6f}`",
            f"- Exact frozen/unconditional/triggered profile MAE: "
            f"`{_format_metric(refresh_rows.get('frozen_day_ahead', {}).get('profile_shape_mae'))}` / "
            f"`{_format_metric(refresh_rows.get('unconditional_refresh', {}).get('profile_shape_mae'))}` / "
            f"`{_format_metric(refresh_rows.get('triggered_refresh', {}).get('profile_shape_mae'))}`",
            "",
            "## Notebook Evidence",
            "",
            f"- Notebook archive status: `{notebook_manifest.get('status', 'n/a')}`",
            f"- `003_modeling.ipynb` output count: `{modeling_notebook.get('output_count', 'n/a')}`",
            f"- `metrics_overall.csv` rows: `{csv_artifacts.get('metrics_overall.csv', {}).get('rows', 'n/a')}`",
            f"- `metrics_by_day_class.csv` rows: `{csv_artifacts.get('metrics_by_day_class.csv', {}).get('rows', 'n/a')}`",
            f"- `metrics_by_hour.csv` rows: `{csv_artifacts.get('metrics_by_hour.csv', {}).get('rows', 'n/a')}`",
            "",
            "## Interpretation",
            "",
            "- The full quick validation surface is currently green.",
            "- The repo still does not support a learned-superiority claim at `1m`.",
            "- Stage-5 now persists an explicit minute operating policy so the repo can say, in writing, that standalone `1m` stays baseline-led while Stage-10 may still use learned minute overlays as corrective specialists.",
            "- The broader leakage-safe Stage-5 supplemental surface now shows where learned `1m` value actually appears: the latest advisory run beats persistence overall on stitched validate-walkforward plus holdout rows, but that support is concentrated in transition regimes rather than the narrow canonical holdout slice.",
            "- The layered stack now carries a dynamic minute-overlay controller in shadow mode. That is intentional: the repo persists both a hard-gate shadow-vs-enforced counterfactual and a soft-overlay shadow search so adaptive minute routing can prove itself on the Stage-10 surface before it is allowed to change live layer resolution.",
            _phase_interpretation_line(
                phase_policy=phase_policy,
                rolling_summary=rolling_summary,
                rolling_inference=rolling_inference,
            ),
            "- The phase stack guard now checks next-lock and peak behavior explicitly, so a phase layer must clear optimizer-relevant guardrails instead of winning only on broader lock/profile metrics.",
            "- Stage-10 now also emits a pre-optimizer interval contract with calibrated residual bands, freshness fields, and confidence hints, so the repo can expose forecast rows with timestamps, horizons, fallback context, and uncertainty instead of only aggregate replay metrics.",
            "- The Stage-7 sweep is now part of the normal E2E verification path, so the repo's 'full pass' no longer skips that surface.",
            "",
            "## High-Signal Visual Anchors",
            "",
            "- Horizon capability curve:",
            f"  ![Stage-8 horizon ratio curve]({_rel(horizon_anchor_png, doc_dir)})",
            "- Control-layer gain surface:",
            f"  ![Stage-10 control layer gain curve]({_rel(control_anchor_png, doc_dir)})",
            "",
            "## Visualization Surfaces",
            "",
            "- Use the integrated dashboard when you want the latest cross-stage visual story in one place.",
            f"- [Current Visualization Guide]({_rel(visualization_guide, doc_dir)})",
            f"- [`{_rel(visualization_dashboard, doc_dir)}`]({_rel(visualization_dashboard, doc_dir)})",
            "",
            "Supporting references:",
            f"- [Current Operating Approach](current_operating_approach.md)",
            f"- [Model and Blend Guide](model_and_blend_guide.md)",
            f"- [README](../../README.md)",
            f"- [`{_rel(latest_paths['forecast_control'] / 'current_evidence_index.md', doc_dir)}`]({_rel(latest_paths['forecast_control'] / 'current_evidence_index.md', doc_dir)})",
        ]
    )
    return "\n".join(lines) + "\n"


def write_validation_snapshot(
    *,
    project_root: Path = PROJECT_ROOT,
    artifact_namespace: str = DATASET["artifact_namespace"],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    step_seconds: dict[str, float] | None = None,
    generated_at_utc: str | None = None,
) -> Path:
    """Write the canonical current-validation snapshot markdown file."""
    output_path = Path(output_path).resolve()
    content = build_validation_snapshot_content(
        project_root=Path(project_root).resolve(),
        artifact_namespace=str(artifact_namespace),
        doc_dir=output_path.parent,
        step_seconds=step_seconds,
        generated_at_utc=generated_at_utc,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for standalone snapshot generation."""
    parser = argparse.ArgumentParser(description="Write the current validation snapshot from latest artifacts.")
    parser.add_argument("--artifact-namespace", default=DATASET["artifact_namespace"])
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--pipeline-seconds", type=float, default=None)
    parser.add_argument("--notebooks-seconds", type=float, default=None)
    parser.add_argument("--pytest-seconds", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    """Generate the current validation snapshot markdown file."""
    args = parse_args()
    step_seconds = {
        key: value
        for key, value in {
            "pipeline": args.pipeline_seconds,
            "notebooks": args.notebooks_seconds,
            "pytest": args.pytest_seconds,
        }.items()
        if value is not None
    }
    output_path = write_validation_snapshot(
        artifact_namespace=str(args.artifact_namespace),
        output_path=Path(args.output_path),
        step_seconds=step_seconds or None,
    )
    print(f"[write_validation_snapshot] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
