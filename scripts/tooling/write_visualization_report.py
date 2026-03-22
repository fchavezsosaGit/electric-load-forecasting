"""Generate an integrated visualization dashboard from the latest repo artifacts.

This tool complements the canonical markdown snapshot by focusing on:

- the goals the repo is trying to satisfy
- which metrics are authoritative at each decision layer
- which visuals answer which modeling or operating question
- the latest evidence across Stage-4, Stage-5, Stage-6, Stage-7, Stage-8, and
  Stage-10 outputs
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import sys

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import DATASET  # noqa: E402

DEFAULT_DOC_OUTPUT_PATH = PROJECT_ROOT / "docs" / "003_modeling" / "current_visualization_guide.md"
DEFAULT_HTML_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / DATASET["artifact_namespace"]
    / "latest"
    / "validation_dashboard.html"
)

_SELECTION_TARGET_FIELDS: dict[str, tuple[str, str, str]] = {
    "endpoint_mae": (
        "learned_endpoint_mae",
        "persistence_endpoint_mae",
        "best_baseline_endpoint_mae",
    ),
    "path_mae": (
        "learned_path_mae",
        "persistence_path_mae",
        "best_baseline_path_mae",
    ),
    "phase_mean_mae": (
        "learned_phase_mean_mae",
        "persistence_phase_mean_mae",
        "best_baseline_phase_mae",
    ),
    "next_lock_mae": (
        "learned_next_lock_mae",
        "persistence_next_lock_mae",
        "best_baseline_next_lock_mae",
    ),
    "profile_shape_mae": (
        "learned_profile_shape_mae",
        "persistence_profile_shape_mae",
        "best_baseline_profile_shape_mae",
    ),
}

_LAYER_PREFIX_TO_LABEL = {
    "day_ahead": "Frozen day-ahead",
    "hourly": "After hourly",
    "phase": "After phase",
    "nowcast": "After nowcast",
}
_LAYER_ORDER = ["day_ahead_frozen", "after_hourly_updates", "after_phase_updates", "after_nowcast_updates"]
_LAYER_DISPLAY = {
    "day_ahead_frozen": "Frozen day-ahead",
    "after_hourly_updates": "After hourly",
    "after_phase_updates": "After phase",
    "after_nowcast_updates": "After nowcast",
}
_STATUS_CLASS = {
    "pass": "status-pass",
    "mixed": "status-mixed",
    "needs_work": "status-needs-work",
}


@dataclass(frozen=True)
class VisualizationReportPaths:
    """Paths written by the integrated visualization report generator."""

    guide_path: Path
    dashboard_path: Path


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


def _supplemental_surface_high_signal_note(segment_evaluation: pd.DataFrame) -> str:
    """Return the strongest learned-positive supplemental segment caption, if any."""
    if segment_evaluation.empty:
        return ""
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
        return ""
    best = candidates.iloc[0]
    return (
        f" Strongest broader segment: {best.get('segment_column', 'segment')}="
        f"{best.get('segment_value', 'unknown')} ({_fmt_metric(best.get('_ratio'))} ratio)."
    )


def _rel(path: Path, base_dir: Path) -> str:
    """Render a filesystem path relative to another directory using POSIX separators."""
    return Path(os.path.relpath(path.resolve(), start=base_dir.resolve())).as_posix()


def _safe_float(value: Any) -> float:
    """Return a stable float or `nan` when conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt_metric(value: Any) -> str:
    """Format a numeric metric for human-readable markdown and HTML tables."""
    numeric = _safe_float(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.3f}"


def _fmt_pct(value: Any) -> str:
    """Format a numeric percentage value."""
    numeric = _safe_float(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.3f}%"


def _truncate_label(value: str, *, limit: int = 52) -> str:
    """Shorten verbose model labels while preserving useful prefixes."""
    text = str(value)
    if len(text) <= int(limit):
        return text
    return f"{text[: int(limit) - 3]}..."


def _candidate_tone(candidate_type: str, candidate_label: str) -> str:
    """Map a candidate row onto one dashboard color family."""
    if str(candidate_label) == "persistence":
        return "#0f172a"
    if "baseline" in str(candidate_type):
        return "#94a3b8"
    return "#0ea5e9"


def _latest_paths(project_root: Path, artifact_namespace: str) -> dict[str, Path]:
    """Return the stable artifact locations used by the integrated report."""
    return {
        "modeling": project_root / "outputs" / "004_modeling" / artifact_namespace,
        "performance": project_root / "outputs" / "005_performance" / artifact_namespace / "latest",
        "multires": project_root / "outputs" / "006_multires" / artifact_namespace / "latest",
        "rollout": project_root / "outputs" / "007_rollout" / artifact_namespace / "latest",
        "rollout_sweep": project_root / "outputs" / "007_rollout" / artifact_namespace / "challenger_sweeps" / "latest",
        "notebooks": project_root / "outputs" / "008_notebook_runs" / artifact_namespace / "latest",
        "horizon_curve": project_root / "outputs" / "009_horizon_curve" / artifact_namespace / "latest",
        "forecast_control": project_root / "outputs" / "010_forecast_control" / artifact_namespace / "latest",
    }


def _selection_metric_fields(selection_target: str) -> tuple[str, str, str]:
    """Resolve the learned, persistence, and best-baseline fields for one objective."""
    return _SELECTION_TARGET_FIELDS.get(
        str(selection_target),
        ("learned_path_mae", "persistence_path_mae", "best_baseline_path_mae"),
    )


def _build_holdout_figure(holdout: pd.DataFrame) -> go.Figure | None:
    """Build the Stage-5 holdout leaderboard figure."""
    if holdout.empty or "mae" not in holdout.columns:
        return None

    sorted_holdout = holdout.sort_values(["mae", "candidate_label"], ascending=[True, True], kind="stable").copy()
    learned = sorted_holdout.loc[
        ~sorted_holdout.get("candidate_type", pd.Series(dtype="string")).astype("string").str.contains(
            "baseline", case=False, na=False
        )
    ].head(4)
    baselines = sorted_holdout.loc[
        sorted_holdout.get("candidate_type", pd.Series(dtype="string")).astype("string").str.contains(
            "baseline", case=False, na=False
        )
    ].head(4)
    persistence = sorted_holdout.loc[
        sorted_holdout.get("candidate_label", pd.Series(dtype="string")).astype("string").eq("persistence")
    ].head(1)
    selected = pd.concat([persistence, learned, baselines], ignore_index=True)
    if selected.empty:
        return None
    selected = selected.drop_duplicates(subset=["candidate_label"]).sort_values("mae", kind="stable")
    selected["display_label"] = selected["candidate_label"].map(lambda value: _truncate_label(str(value)))
    selected["tone"] = [
        _candidate_tone(str(row.get("candidate_type", "")), str(row.get("candidate_label", "")))
        for _, row in selected.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=selected["mae"],
            y=selected["display_label"],
            orientation="h",
            marker={"color": selected["tone"], "line": {"color": "#e2e8f0", "width": 0.8}},
            customdata=np.column_stack(
                [
                    selected["candidate_label"].astype(str),
                    selected.get("candidate_type", pd.Series(index=selected.index, dtype="string")).astype(str),
                    selected.get("mae_pct", pd.Series(index=selected.index, dtype="float64")).astype(float),
                    selected.get(
                        "mae_ratio_to_persistence",
                        pd.Series(index=selected.index, dtype="float64"),
                    ).astype(float),
                ]
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Type: %{customdata[1]}<br>"
                "Holdout MAE: %{x:.3f}<br>"
                "Holdout MAE%%: %{customdata[2]:.3f}%<br>"
                "MAE / persistence: %{customdata[3]:.3f}x<extra></extra>"
            ),
            name="Holdout MAE",
        )
    )
    if not persistence.empty:
        fig.add_vline(
            x=float(persistence.iloc[0]["mae"]),
            line_dash="dot",
            line_color="#ef4444",
            annotation_text="Persistence",
            annotation_position="top right",
        )

    fig.update_layout(
        template="plotly_white",
        height=420,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="Holdout MAE (native load units)",
        yaxis_title="",
    )
    return fig


def _build_multires_figure(multires_metrics: pd.DataFrame) -> go.Figure | None:
    """Build the Stage-6 runtime-vs-gain tradeoff figure."""
    if multires_metrics.empty:
        return None

    candidates = multires_metrics.loc[
        multires_metrics.get("comparison_mode", pd.Series(dtype="string")).astype("string").eq("matched_horizon")
        & multires_metrics.get("candidate_type", pd.Series(dtype="string")).astype("string").eq("learned")
        & pd.to_numeric(multires_metrics.get("runtime_seconds"), errors="coerce").gt(0.0)
    ].copy()
    if candidates.empty:
        return None

    candidates["runtime_seconds"] = pd.to_numeric(candidates["runtime_seconds"], errors="coerce")
    candidates["mae_ratio_to_persistence"] = pd.to_numeric(
        candidates["mae_ratio_to_persistence"],
        errors="coerce",
    )
    candidates["horizon_minutes"] = pd.to_numeric(candidates["horizon_minutes"], errors="coerce")
    candidates = candidates.dropna(subset=["runtime_seconds", "mae_ratio_to_persistence", "horizon_minutes"])
    if candidates.empty:
        return None

    resolution_colors = {
        resolution: color
        for resolution, color in zip(
            sorted(candidates["resolution"].astype(str).unique()),
            ["#0ea5e9", "#14b8a6", "#f97316", "#ef4444", "#8b5cf6", "#64748b"],
            strict=False,
        )
    }
    symbol_map = {
        "recursive": "circle",
        "direct_endpoint": "diamond",
        "path_baseline": "square",
        "raw": "triangle-up",
    }

    fig = go.Figure()
    for resolution in sorted(candidates["resolution"].astype(str).unique()):
        subset = candidates.loc[candidates["resolution"].astype("string").eq(resolution)].copy()
        fig.add_trace(
            go.Scatter(
                x=subset["runtime_seconds"],
                y=subset["mae_ratio_to_persistence"],
                mode="markers",
                name=str(resolution),
                marker={
                    "color": resolution_colors[resolution],
                    "size": np.clip(8.0 + subset["horizon_minutes"].to_numpy(dtype=float) / 20.0, 9.0, 28.0),
                    "opacity": 0.88,
                    "symbol": [
                        symbol_map.get(str(strategy), "circle")
                        for strategy in subset.get("forecast_strategy", pd.Series(dtype="string")).astype(str)
                    ],
                    "line": {
                        "color": [
                            "#0f172a" if bool(value) else "#ffffff"
                            for value in subset.get("eligible", pd.Series(dtype="bool")).fillna(False)
                        ],
                        "width": [
                            1.4 if bool(value) else 0.6
                            for value in subset.get("eligible", pd.Series(dtype="bool")).fillna(False)
                        ],
                    },
                },
                customdata=np.column_stack(
                    [
                        subset["resolution"].astype(str),
                        subset["horizon_minutes"].map(lambda value: f"{float(value):.0f}m"),
                        subset.get("feature_set", pd.Series(index=subset.index, dtype="string")).astype(str),
                        subset.get("model_label", pd.Series(index=subset.index, dtype="string")).astype(str),
                        subset.get("forecast_strategy", pd.Series(index=subset.index, dtype="string")).astype(str),
                        subset.get("eval_coverage", pd.Series(index=subset.index, dtype="float64")).astype(float),
                        subset.get("eligible", pd.Series(index=subset.index, dtype="bool")).astype(bool),
                    ]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]} / %{customdata[1]}</b><br>"
                    "Feature set: %{customdata[2]}<br>"
                    "Model: %{customdata[3]}<br>"
                    "Strategy: %{customdata[4]}<br>"
                    "Runtime: %{x:.3f}s<br>"
                    "MAE / persistence: %{y:.3f}x<br>"
                    "Coverage: %{customdata[5]:.3f}<br>"
                    "Eligible: %{customdata[6]}<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=1.0, line_dash="dot", line_color="#ef4444")
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="Candidate runtime (seconds, log scale)",
        yaxis_title="MAE ratio to persistence",
        legend_title="Resolution",
    )
    fig.update_xaxes(type="log")
    return fig


def _build_horizon_ratio_figure(horizon: pd.DataFrame) -> go.Figure | None:
    """Build the Stage-8 objective-aware horizon ratio figure."""
    if horizon.empty:
        return None

    ratio_rows: list[dict[str, Any]] = []
    for _, row in horizon.sort_values("horizon_minutes", kind="stable").iterrows():
        learned_field, persistence_field, baseline_field = _selection_metric_fields(str(row.get("selection_target", "")))
        learned_value = _safe_float(row.get(learned_field))
        persistence_value = _safe_float(row.get(persistence_field))
        baseline_value = _safe_float(row.get(baseline_field))
        if not np.isfinite(learned_value) or not np.isfinite(persistence_value):
            continue
        ratio_rows.append(
            {
                "horizon_minutes": _safe_float(row.get("horizon_minutes")),
                "selection_target": str(row.get("selection_target", "")),
                "candidate_label": str(row.get("candidate_label", "")),
                "ratio_to_persistence": learned_value / persistence_value if persistence_value else float("nan"),
                "ratio_to_best_baseline": learned_value / baseline_value if baseline_value else float("nan"),
            }
        )
    ratio_frame = pd.DataFrame(ratio_rows)
    if ratio_frame.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ratio_frame["horizon_minutes"],
            y=ratio_frame["ratio_to_persistence"],
            mode="lines+markers+text",
            line={"color": "#0ea5e9", "width": 3},
            marker={"size": 10, "color": "#0ea5e9"},
            text=ratio_frame["selection_target"],
            textposition="top center",
            name="Learned / persistence",
            customdata=np.column_stack(
                [
                    ratio_frame["candidate_label"].astype(str),
                    ratio_frame["selection_target"].astype(str),
                ]
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Target objective: %{customdata[1]}<br>"
                "Horizon: %{x:.0f}m<br>"
                "Ratio to persistence: %{y:.3f}x<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ratio_frame["horizon_minutes"],
            y=ratio_frame["ratio_to_best_baseline"],
            mode="lines+markers",
            line={"color": "#f97316", "width": 2, "dash": "dash"},
            marker={"size": 9, "color": "#f97316"},
            name="Learned / best baseline",
            customdata=ratio_frame["selection_target"],
            hovertemplate=(
                "Target objective: %{customdata}<br>"
                "Horizon: %{x:.0f}m<br>"
                "Ratio to best baseline: %{y:.3f}x<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="#ef4444")
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="Forecast horizon (minutes)",
        yaxis_title="Relative error ratio",
    )
    return fig


def _build_horizon_matrix_figure(horizon: pd.DataFrame) -> go.Figure | None:
    """Build a persistence win/loss matrix across horizon and objective."""
    if horizon.empty:
        return None

    metric_columns = [
        ("endpoint", "beats_persistence_endpoint"),
        ("path", "beats_persistence_path"),
        ("phase", "beats_persistence_phase"),
        ("next lock", "beats_persistence_next_lock"),
        ("profile shape", "beats_persistence_profile_shape"),
    ]
    ordered = horizon.sort_values("horizon_minutes", kind="stable")
    z_values: list[list[int]] = []
    text_values: list[list[str]] = []
    for _, metric_column in metric_columns:
        row_values: list[int] = []
        row_text: list[str] = []
        for _, row in ordered.iterrows():
            beat_value = bool(row.get(metric_column, False))
            row_values.append(1 if beat_value else 0)
            row_text.append("Win" if beat_value else "Loss")
        z_values.append(row_values)
        text_values.append(row_text)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z_values,
                x=[f"{_safe_float(value):.0f}m" for value in ordered["horizon_minutes"]],
                y=[label for label, _ in metric_columns],
                text=text_values,
                texttemplate="%{text}",
                colorscale=[
                    [0.0, "#fee2e2"],
                    [0.499, "#fee2e2"],
                    [0.5, "#dcfce7"],
                    [1.0, "#16a34a"],
                ],
                showscale=False,
                hovertemplate="Horizon %{x}<br>Metric %{y}<br>Status %{text}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        height=340,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="Forecast horizon",
        yaxis_title="Metric family",
    )
    return fig


def _build_control_layer_figure(
    control_summary: pd.DataFrame,
    rolling_summary: pd.DataFrame,
) -> go.Figure | None:
    """Build the exact-vs-rolling Stage-10 layer trajectory figure."""
    if control_summary.empty and rolling_summary.empty:
        return None

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Exact control evaluation", "Rolling evaluation"),
        horizontal_spacing=0.12,
    )
    scope_frames = [control_summary.copy(), rolling_summary.copy()]
    metric_specs = [
        ("lock_mae", "Lock MAE", "#0ea5e9"),
        ("profile_shape_mae", "Profile-shape MAE", "#f97316"),
    ]
    for col_idx, frame in enumerate(scope_frames, start=1):
        if frame.empty:
            continue
        indexed = frame.set_index("layer")
        x_labels = [_LAYER_DISPLAY[layer] for layer in _LAYER_ORDER if layer in indexed.index]
        for metric_name, trace_name, color in metric_specs:
            y_values = [
                _safe_float(indexed.loc[layer, metric_name])
                for layer in _LAYER_ORDER
                if layer in indexed.index
            ]
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=y_values,
                    mode="lines+markers",
                    name=trace_name,
                    marker={"size": 9, "color": color},
                    line={"width": 3, "color": color},
                    showlegend=col_idx == 1,
                    hovertemplate=(
                        f"{trace_name}<br>"
                        "Layer: %{x}<br>"
                        "Metric: %{y:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        legend={"orientation": "h", "y": 1.12, "x": 0.0},
    )
    fig.update_yaxes(title_text="Error (native load units)", row=1, col=1)
    fig.update_yaxes(title_text="Error (native load units)", row=1, col=2)
    return fig


def _control_by_cycle_long_frame(
    exact_by_cycle: pd.DataFrame,
    rolling_by_cycle: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize exact and rolling by-cycle control outputs into one long frame."""
    long_rows: list[dict[str, Any]] = []
    frame_specs = [
        ("exact_control", exact_by_cycle.copy()),
        ("rolling_evaluation", rolling_by_cycle.copy()),
    ]
    for scope_label, frame in frame_specs:
        if frame.empty:
            continue
        for prefix, layer_label in _LAYER_PREFIX_TO_LABEL.items():
            lock_column = f"{prefix}_lock_mae"
            profile_column = f"{prefix}_profile_shape_mae"
            if lock_column not in frame.columns or profile_column not in frame.columns:
                continue
            for value in pd.to_numeric(frame[lock_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna():
                long_rows.append(
                    {
                        "scope": scope_label,
                        "layer": layer_label,
                        "metric_family": "Lock MAE",
                        "metric_value": float(value),
                    }
                )
            for value in pd.to_numeric(frame[profile_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna():
                long_rows.append(
                    {
                        "scope": scope_label,
                        "layer": layer_label,
                        "metric_family": "Profile-shape MAE",
                        "metric_value": float(value),
                    }
                )
    return pd.DataFrame(long_rows)


def _build_cycle_distribution_figure(
    exact_by_cycle: pd.DataFrame,
    rolling_by_cycle: pd.DataFrame,
) -> go.Figure | None:
    """Build the by-cycle error-distribution figure for Stage-10."""
    long_frame = _control_by_cycle_long_frame(exact_by_cycle, rolling_by_cycle)
    if long_frame.empty:
        return None

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Lock MAE distribution", "Profile-shape MAE distribution"),
        horizontal_spacing=0.12,
    )
    scope_colors = {
        "exact_control": "#0ea5e9",
        "rolling_evaluation": "#f97316",
    }
    for col_idx, metric_family in enumerate(["Lock MAE", "Profile-shape MAE"], start=1):
        metric_subset = long_frame.loc[long_frame["metric_family"].eq(metric_family)]
        if metric_subset.empty:
            continue
        for scope in ["exact_control", "rolling_evaluation"]:
            scope_subset = metric_subset.loc[metric_subset["scope"].eq(scope)]
            if scope_subset.empty:
                continue
            fig.add_trace(
                go.Box(
                    x=scope_subset["layer"],
                    y=scope_subset["metric_value"],
                    name="Exact control" if scope == "exact_control" else "Rolling evaluation",
                    marker_color=scope_colors[scope],
                    boxmean=True,
                    opacity=0.76,
                    legendgroup=scope,
                    showlegend=col_idx == 1,
                    hovertemplate=(
                        f"{'Exact control' if scope == 'exact_control' else 'Rolling evaluation'}<br>"
                        "Layer: %{x}<br>"
                        "Metric: %{y:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        boxmode="group",
        legend={"orientation": "h", "y": 1.12, "x": 0.0},
    )
    fig.update_yaxes(title_text="Error (native load units)", row=1, col=1)
    fig.update_yaxes(title_text="Error (native load units)", row=1, col=2)
    return fig


def _build_refresh_figure(refresh_summary: pd.DataFrame) -> go.Figure | None:
    """Build the day-ahead refresh policy comparison figure."""
    if refresh_summary.empty or "scenario" not in refresh_summary.columns:
        return None

    frame = refresh_summary.copy()
    frame["scenario_label"] = frame["scenario"].astype(str).map(
        {
            "frozen_day_ahead": "Frozen",
            "unconditional_refresh": "Always refresh",
            "triggered_refresh": "Triggered refresh",
        }
    ).fillna(frame["scenario"].astype(str))
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["scenario_label"],
            y=pd.to_numeric(frame.get("lock_mae"), errors="coerce"),
            name="Lock MAE",
            marker_color="#0ea5e9",
            customdata=np.column_stack(
                [
                    pd.to_numeric(frame.get("refresh_update_count"), errors="coerce").fillna(0.0),
                    pd.to_numeric(frame.get("lock_mae_gain_vs_frozen"), errors="coerce").fillna(0.0),
                ]
            ),
            hovertemplate=(
                "Scenario: %{x}<br>"
                "Lock MAE: %{y:.3f}<br>"
                "Refresh count: %{customdata[0]:.3f}<br>"
                "Gain vs frozen: %{customdata[1]:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=frame["scenario_label"],
            y=pd.to_numeric(frame.get("profile_shape_mae"), errors="coerce"),
            name="Profile-shape MAE",
            marker_color="#f97316",
            customdata=np.column_stack(
                [
                    pd.to_numeric(frame.get("refresh_update_count"), errors="coerce").fillna(0.0),
                    pd.to_numeric(frame.get("profile_shape_mae_gain_vs_frozen"), errors="coerce").fillna(0.0),
                ]
            ),
            hovertemplate=(
                "Scenario: %{x}<br>"
                "Profile-shape MAE: %{y:.3f}<br>"
                "Refresh count: %{customdata[0]:.3f}<br>"
                "Gain vs frozen: %{customdata[1]:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=380,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        barmode="group",
        xaxis_title="Refresh policy",
        yaxis_title="Error (native load units)",
    )
    return fig


def _table_html(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """Render a compact HTML table from row dictionaries."""
    if not rows:
        return "<p class='empty-state'>No table rows were available from the latest artifacts.</p>"
    header_html = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body_rows: list[str] = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(row.get(key, '')))}</td>"
            for key, _ in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        "<div class='table-shell'>"
        "<table class='data-table'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _table_markdown(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """Render a stable markdown table without optional dependencies."""
    if not rows:
        return "_No table rows were available from the latest artifacts._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for row in rows:
        values = []
        for key, _ in columns:
            cell = str(row.get(key, "")).replace("|", "\\|")
            values.append(cell)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _integrated_visual_specs() -> list[dict[str, str]]:
    """Return the integrated visual catalog used in the dashboard and guide."""
    return [
        {
            "id": "holdout_leaderboard",
            "title": "1-minute Holdout Leaderboard",
            "purpose": "Answer the most honest deployment question first: whether any learned 1-minute challenger actually beats persistence on holdout.",
            "how_to_read": "Lower bars are better. The persistence reference line marks the baseline the learned model must beat before it should be promoted.",
            "look_for": "A deployable learned winner should sit left of persistence on raw MAE, not only look good on validation or blended diagnostics.",
            "sources": "Stage-5 holdout_evaluation.csv and deployment_recommendation.json",
        },
        {
            "id": "multires_tradeoff",
            "title": "Matched-Horizon Runtime vs Persistence",
            "purpose": "Show which Stage-6 learned candidates earn their compute cost once horizon, runtime, and persistence-relative error are considered together.",
            "how_to_read": "Lower is better on the y-axis because ratios below 1.0 beat persistence. Moving left is better on runtime, and larger markers indicate longer horizons.",
            "look_for": "Candidates near the lower-left frontier and, ideally, below parity. Fast candidates that still lose to persistence are not operational wins.",
            "sources": "Stage-6 matched_horizon_metrics.csv",
        },
        {
            "id": "horizon_ratio_curve",
            "title": "Objective-Aware Horizon Ratio Curve",
            "purpose": "Summarize where the learned stack beats persistence or the best baseline once the metric is matched to the horizon's real operating objective.",
            "how_to_read": "Values below 1.0 are wins. The blue line compares learned winners to persistence; the orange dashed line compares them to the strongest baseline.",
            "look_for": "Parity crossings, especially where short horizons win on next-lock quality while day-ahead horizons win on profile shape instead.",
            "sources": "Stage-8 horizon_curve_summary.csv",
        },
        {
            "id": "horizon_win_matrix",
            "title": "Horizon Win Matrix",
            "purpose": "Separate 'where learned helps' from 'where learned looks good only on the wrong metric' by showing win/loss status across metric families.",
            "how_to_read": "Green cells mean the learned winner beats persistence for that horizon-metric pair. Red cells mean persistence still wins.",
            "look_for": "Concentrated green on the metric family that actually matters for that horizon, not blanket wins across every metric.",
            "sources": "Stage-8 horizon_curve_summary.csv",
        },
        {
            "id": "control_layer_trajectory",
            "title": "Control-Layer Error Trajectory",
            "purpose": "Show whether each successive forecast update layer improves the exact control cycle and the broader rolling evaluation surface.",
            "how_to_read": "The lines should move down as the stack progresses from frozen day-ahead to hourly, phase, and nowcast corrections.",
            "look_for": "Consistent downward error movement in both exact and rolling scopes, not just a one-window improvement.",
            "sources": "Stage-10 control_backtest_summary.csv and rolling_control_backtest_summary.csv",
        },
        {
            "id": "control_cycle_distribution",
            "title": "Cycle-by-Cycle Error Distribution",
            "purpose": "Expose variability, not only averages, so control-layer gains can be judged on stability and tail behavior.",
            "how_to_read": "Each box shows the distribution of cycle-level errors for one layer. Lower medians and tighter boxes indicate more reliable operating behavior.",
            "look_for": "Later layers should shift the full distribution down, not only improve the mean while leaving heavy tails intact.",
            "sources": "Stage-10 control_backtest_by_cycle.csv and rolling_control_backtest_by_cycle.csv",
        },
        {
            "id": "refresh_policy_comparison",
            "title": "Day-Ahead Refresh Policy Comparison",
            "purpose": "Explain why the chosen day-ahead refresh policy is frozen, always-refresh, or triggered-refresh rather than treating refresh as a hidden rule.",
            "how_to_read": "Lower bars are better. Compare frozen, unconditional refresh, and triggered refresh on lock MAE and profile-shape MAE together.",
            "look_for": "Triggered refresh should capture most of the unconditional improvement without forcing unnecessary updates every cycle.",
            "sources": "Stage-10 day_ahead_refresh_summary.csv",
        },
    ]


def _metric_catalog_rows() -> list[dict[str, str]]:
    """Return the repo-wide metric glossary used by the dashboard."""
    return [
        {
            "metric": "MAE",
            "use_case": "Native-unit average error for one facility.",
            "authoritative_scope": "Stage-4 and Stage-5 holdout decisions.",
            "good_pattern": "Lower than persistence on holdout and stable across folds.",
            "warning": "Can hide whether the error is concentrated in expensive transition windows.",
        },
        {
            "metric": "MAE%",
            "use_case": "Scale-normalized comparison across horizons and runs.",
            "authoritative_scope": "Cross-horizon reporting and cross-resolution comparisons.",
            "good_pattern": "Moves down with raw MAE when comparing across different load scales.",
            "warning": "Do not treat it as a substitute for native-unit operating budgets.",
        },
        {
            "metric": "RMSE",
            "use_case": "Penalty on large misses and transition spikes.",
            "authoritative_scope": "Stage-4 and Stage-5 transition-risk checks.",
            "good_pattern": "Improves with MAE when the model is not trading off mean error for larger outliers.",
            "warning": "A lower RMSE alone does not justify deployment if MAE still loses to persistence.",
        },
        {
            "metric": "MAE ratio to persistence",
            "use_case": "Simple parity test against the operational baseline.",
            "authoritative_scope": "Stage-6 and Stage-8 cross-horizon screening.",
            "good_pattern": "Below 1.0.",
            "warning": "Only compare the ratio on the metric that matters for the decision layer.",
        },
        {
            "metric": "next_lock_mae",
            "use_case": "Error on the next locked interval after an update.",
            "authoritative_scope": "Short corrective horizons such as 15m and 60m.",
            "good_pattern": "Large reductions versus persistence and strong phase/hourly corrections.",
            "warning": "A candidate can win next-lock MAE while still miss whole-day profile quality.",
        },
        {
            "metric": "profile_shape_mae",
            "use_case": "Shape quality after rescaling predicted energy to actual energy.",
            "authoritative_scope": "Day-ahead profile selection and refresh policy decisions.",
            "good_pattern": "Lower than persistence when total-energy bias is removed.",
            "warning": "Do not use it alone when the business cost is tied to the next locked interval.",
        },
        {
            "metric": "lock_mae",
            "use_case": "Operationally costly locked-interval error in the control backtest.",
            "authoritative_scope": "Stage-10 control-layer evaluation.",
            "good_pattern": "Each later layer reduces it materially and the gain remains visible on rolling evaluation.",
            "warning": "Exact-window wins are not enough if rolling confidence intervals are flat.",
        },
        {
            "metric": "energy_mae",
            "use_case": "Total energy mismatch over the control horizon.",
            "authoritative_scope": "Stage-10 whole-profile sanity checks.",
            "good_pattern": "Falls alongside profile-shape MAE for profile-oriented improvements.",
            "warning": "A low energy miss can still hide poor minute-level timing.",
        },
    ]


def _stage_question_rows() -> list[dict[str, str]]:
    """Return the stage-question map for the dashboard and guide."""
    return [
        {
            "stage": "Stage-4 notebook benchmark",
            "question": "What does the core 1-minute modeling surface look like before promotion or control replay?",
            "authoritative_metric": "MAE, RMSE, error-by-hour structure",
            "main_visual": "Existing Stage-4 PNG gallery",
        },
        {
            "stage": "Stage-5 holdout gate",
            "question": "Should a learned 1-minute model replace persistence on holdout right now?",
            "authoritative_metric": "Holdout MAE plus bootstrap support versus persistence",
            "main_visual": "1-minute Holdout Leaderboard",
        },
        {
            "stage": "Stage-6 multiresolution",
            "question": "Which matched-horizon candidates earn their runtime cost when compared to persistence?",
            "authoritative_metric": "MAE ratio to persistence, coverage, runtime",
            "main_visual": "Matched-Horizon Runtime vs Persistence",
        },
        {
            "stage": "Stage-7 rollout selection",
            "question": "Which rollout policy wins once the objective changes from endpoint error to path, phase, or profile quality?",
            "authoritative_metric": "Selection-target metric by objective",
            "main_visual": "Stage-7 winner table",
        },
        {
            "stage": "Stage-8 horizon curve",
            "question": "At which horizons does the learned stack beat persistence on the right metric?",
            "authoritative_metric": "Objective-matched ratio to persistence",
            "main_visual": "Objective-Aware Horizon Ratio Curve + Horizon Win Matrix",
        },
        {
            "stage": "Stage-10 forecast control",
            "question": "Do layered updates reduce lock error and profile error on exact and rolling control cycles?",
            "authoritative_metric": "lock_mae, profile_shape_mae, rolling gain confidence",
            "main_visual": "Control-Layer Trajectory, Cycle Distribution, and Refresh Policy Comparison",
        },
    ]


def _manifest_health_rows(manifests: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """Normalize the latest manifest statuses into table rows."""
    stage_labels = {
        "modeling": "Stage-4 notebook benchmark",
        "performance": "Stage-5 holdout gate",
        "multires": "Stage-6 multires",
        "rollout": "Stage-7 rollout",
        "rollout_sweep": "Stage-7 challenger sweep",
        "horizon_curve": "Stage-8 horizon curve",
        "forecast_control": "Stage-10 forecast control",
        "notebooks": "Notebook archive",
    }
    rows: list[dict[str, str]] = []
    for stage_key, stage_label in stage_labels.items():
        manifest = manifests.get(stage_key) or {}
        status = str(manifest.get("status", "success" if manifest else "missing"))
        rows.append(
            {
                "stage": stage_label,
                "status": status,
                "run_id": str(manifest.get("run_id", "n/a")),
            }
        )
    return rows


def _goal_rows(
    *,
    manifests: dict[str, dict[str, Any]],
    holdout: pd.DataFrame,
    deployment_recommendation: dict[str, Any],
    horizon: pd.DataFrame,
    control_summary: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    rolling_inference: pd.DataFrame,
) -> list[dict[str, str]]:
    """Build the latest-state goal and success rows."""
    rows: list[dict[str, str]] = []

    persistence_row = holdout.loc[
        holdout.get("candidate_label", pd.Series(dtype="string")).astype("string").eq("persistence")
    ]
    learned_rows = holdout.loc[
        ~holdout.get("candidate_type", pd.Series(dtype="string")).astype("string").str.contains(
            "baseline",
            case=False,
            na=False,
        )
    ].sort_values("mae", kind="stable")
    learned_row = learned_rows.iloc[0] if not learned_rows.empty else pd.Series(dtype=object)
    persistence_mae = _safe_float(persistence_row.iloc[0]["mae"]) if not persistence_row.empty else float("nan")
    learned_mae = _safe_float(learned_row.get("mae"))
    stage5_status = (
        "pass"
        if str(deployment_recommendation.get("recommended_candidate_label", "")) != "persistence"
        and np.isfinite(learned_mae)
        and np.isfinite(persistence_mae)
        and learned_mae < persistence_mae
        else "needs_work"
    )
    gap = learned_mae - persistence_mae
    rows.append(
        {
            "goal": "Honest 1-minute deployability",
            "decision_layer": "Stage-5 holdout gate",
            "primary_metric": "Holdout MAE versus persistence",
            "current_readout": (
                f"Best learned {learned_row.get('candidate_label', 'n/a')} "
                f"at {_fmt_metric(learned_mae)} vs persistence {_fmt_metric(persistence_mae)} "
                f"(gap {_fmt_metric(gap)})."
            ),
            "success_looks_like": "A learned candidate beats persistence on holdout, not only on validation or blended diagnostics.",
            "status": stage5_status,
            "status_label": "On track" if stage5_status == "pass" else "Not yet met",
        }
    )

    horizon_wins = horizon.loc[
        (
            pd.to_numeric(horizon.get("horizon_minutes"), errors="coerce").gt(1.0)
        )
        & (
            horizon.apply(
                lambda row: bool(
                    row.get(
                        {
                            "endpoint_mae": "beats_persistence_endpoint",
                            "path_mae": "beats_persistence_path",
                            "phase_mean_mae": "beats_persistence_phase",
                            "next_lock_mae": "beats_persistence_next_lock",
                            "profile_shape_mae": "beats_persistence_profile_shape",
                        }.get(str(row.get("selection_target", "")), "beats_persistence_path"),
                        False,
                    )
                ),
                axis=1,
            )
        )
    ].copy()
    horizon_status = "pass" if not horizon_wins.empty else "needs_work"
    horizon_summary = ", ".join(
        f"{int(_safe_float(row['horizon_minutes']))}m on {row['selection_target']}"
        for _, row in horizon_wins.iterrows()
    ) or "No learned horizon currently beats persistence on its selected target."
    rows.append(
        {
            "goal": "Objective-aware horizon coverage",
            "decision_layer": "Stage-8 horizon curve",
            "primary_metric": "Ratio to persistence on the selected horizon objective",
            "current_readout": f"Learned wins currently appear at {horizon_summary}.",
            "success_looks_like": "Short horizons win on short-horizon objectives and day-ahead horizons win on profile objectives.",
            "status": horizon_status,
            "status_label": "Strong signal" if horizon_status == "pass" else "Needs stronger evidence",
        }
    )

    control_exact = control_summary.set_index("layer") if not control_summary.empty else pd.DataFrame()
    control_rolling = rolling_summary.set_index("layer") if not rolling_summary.empty else pd.DataFrame()
    exact_start = _safe_float(control_exact.get("lock_mae", pd.Series(dtype="float64")).get("day_ahead_frozen"))
    exact_end = _safe_float(control_exact.get("lock_mae", pd.Series(dtype="float64")).get("after_nowcast_updates"))
    rolling_start = _safe_float(control_rolling.get("lock_mae", pd.Series(dtype="float64")).get("day_ahead_frozen"))
    rolling_end = _safe_float(control_rolling.get("lock_mae", pd.Series(dtype="float64")).get("after_nowcast_updates"))
    exact_reduction = 100.0 * (exact_start - exact_end) / exact_start if exact_start else float("nan")
    rolling_reduction = 100.0 * (rolling_start - rolling_end) / rolling_start if rolling_start else float("nan")
    phase_inference = rolling_inference.loc[
        rolling_inference.get("scope", pd.Series(dtype="string")).astype("string").eq("rolling_evaluation")
        & rolling_inference.get("comparison_label", pd.Series(dtype="string")).astype("string").eq("phase_vs_hourly")
        & rolling_inference.get("metric_name", pd.Series(dtype="string")).astype("string").eq("lock_mae")
    ]
    phase_supported = False
    if not phase_inference.empty:
        inference_row = phase_inference.iloc[0]
        phase_supported = bool(inference_row.get("candidate_better_than_baseline", False)) and bool(
            inference_row.get("gain_ci_excludes_zero", False)
        )
    exact_hourly_lock = _safe_float(control_exact.get("lock_mae", pd.Series(dtype="float64")).get("after_hourly_updates"))
    exact_phase_lock = _safe_float(control_exact.get("lock_mae", pd.Series(dtype="float64")).get("after_phase_updates"))
    exact_hourly_profile = _safe_float(
        control_exact.get("profile_shape_mae", pd.Series(dtype="float64")).get("after_hourly_updates")
    )
    exact_phase_profile = _safe_float(
        control_exact.get("profile_shape_mae", pd.Series(dtype="float64")).get("after_phase_updates")
    )
    rolling_hourly_lock = _safe_float(
        control_rolling.get("lock_mae", pd.Series(dtype="float64")).get("after_hourly_updates")
    )
    rolling_phase_lock = _safe_float(
        control_rolling.get("lock_mae", pd.Series(dtype="float64")).get("after_phase_updates")
    )
    rolling_hourly_profile = _safe_float(
        control_rolling.get("profile_shape_mae", pd.Series(dtype="float64")).get("after_hourly_updates")
    )
    rolling_phase_profile = _safe_float(
        control_rolling.get("profile_shape_mae", pd.Series(dtype="float64")).get("after_phase_updates")
    )
    phase_passthrough = all(
        np.isfinite(left)
        and np.isfinite(right)
        and np.isclose(left, right, atol=1e-9, rtol=0.0)
        for left, right in (
            (exact_phase_lock, exact_hourly_lock),
            (exact_phase_profile, exact_hourly_profile),
            (rolling_phase_lock, rolling_hourly_lock),
            (rolling_phase_profile, rolling_hourly_profile),
        )
    )
    control_status = (
        "pass"
        if np.isfinite(exact_reduction) and exact_reduction > 50.0 and np.isfinite(rolling_reduction) and rolling_reduction > 50.0
        else "mixed"
    )
    if phase_passthrough:
        control_phase_text = (
            "The current phase slot resolves to hourly passthrough after the broader "
            "rolling-support guard, so the measured stack gain currently comes from "
            "the hourly and nowcast layers."
        )
    elif phase_supported:
        control_phase_text = "Phase gain is statistically supported on rolling evaluation."
    else:
        control_phase_text = "Phase gain is present but still needs more support on rolling evaluation."
    rows.append(
        {
            "goal": "Layered control-stack improvement",
            "decision_layer": "Stage-10 forecast control",
            "primary_metric": "Lock MAE and profile-shape MAE through the stacked layers",
            "current_readout": (
                f"Exact lock error falls {_fmt_pct(exact_reduction)} and rolling lock error falls {_fmt_pct(rolling_reduction)} "
                f"from frozen day-ahead to the final nowcast layer. {control_phase_text}"
            ),
            "success_looks_like": "Later layers should reduce both exact-window and rolling control error, not just one of them.",
            "status": control_status,
            "status_label": "Operationally strong" if control_status == "pass" else "Mixed but improving",
        }
    )

    manifest_rows = _manifest_health_rows(manifests)
    success_count = sum(1 for row in manifest_rows if row["status"] == "success")
    validation_status = "pass" if success_count == len(manifest_rows) else "mixed"
    rows.append(
        {
            "goal": "Reproducible evidence chain",
            "decision_layer": "End-to-end validation surface",
            "primary_metric": "Latest artifact and notebook manifest health",
            "current_readout": f"{success_count}/{len(manifest_rows)} latest manifests report success.",
            "success_looks_like": "The reporting layer is trustworthy only when the underlying artifact chain is green.",
            "status": validation_status,
            "status_label": "Green" if validation_status == "pass" else "Check upstream artifacts",
        }
    )

    return rows


def _summary_cards(
    *,
    manifests: dict[str, dict[str, Any]],
    deployment_recommendation: dict[str, Any],
    holdout: pd.DataFrame,
    holdout_coverage_summary: dict[str, Any],
    supplemental_surface_advisory: dict[str, Any],
    supplemental_surface_segment_evaluation: pd.DataFrame,
    horizon: pd.DataFrame,
    control_summary: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    runtime_summary: dict[str, Any],
) -> list[dict[str, str]]:
    """Build the hero-card metrics for the dashboard header."""
    persistence_row = holdout.loc[
        holdout.get("candidate_label", pd.Series(dtype="string")).astype("string").eq("persistence")
    ]
    learned_rows = holdout.loc[
        ~holdout.get("candidate_type", pd.Series(dtype="string")).astype("string").str.contains(
            "baseline",
            case=False,
            na=False,
        )
    ].sort_values("mae", kind="stable")
    learned_row = learned_rows.iloc[0] if not learned_rows.empty else pd.Series(dtype=object)
    persistence_mae = _safe_float(persistence_row.iloc[0]["mae"]) if not persistence_row.empty else float("nan")
    learned_mae = _safe_float(learned_row.get("mae"))
    gap = learned_mae - persistence_mae

    horizon_win_count = 0
    if not horizon.empty:
        for _, row in horizon.iterrows():
            metric_column = {
                "endpoint_mae": "beats_persistence_endpoint",
                "path_mae": "beats_persistence_path",
                "phase_mean_mae": "beats_persistence_phase",
                "next_lock_mae": "beats_persistence_next_lock",
                "profile_shape_mae": "beats_persistence_profile_shape",
            }.get(str(row.get("selection_target", "")), "beats_persistence_path")
            if bool(row.get(metric_column, False)):
                horizon_win_count += 1

    exact = control_summary.set_index("layer") if not control_summary.empty else pd.DataFrame()
    rolling = rolling_summary.set_index("layer") if not rolling_summary.empty else pd.DataFrame()
    exact_start = _safe_float(exact.get("lock_mae", pd.Series(dtype="float64")).get("day_ahead_frozen"))
    exact_end = _safe_float(exact.get("lock_mae", pd.Series(dtype="float64")).get("after_nowcast_updates"))
    rolling_start = _safe_float(rolling.get("lock_mae", pd.Series(dtype="float64")).get("day_ahead_frozen"))
    rolling_end = _safe_float(rolling.get("lock_mae", pd.Series(dtype="float64")).get("after_nowcast_updates"))
    exact_reduction = 100.0 * (exact_start - exact_end) / exact_start if exact_start else float("nan")
    rolling_reduction = 100.0 * (rolling_start - rolling_end) / rolling_start if rolling_start else float("nan")

    manifest_rows = _manifest_health_rows(manifests)
    success_count = sum(1 for row in manifest_rows if row["status"] == "success")
    holdout_caption = f"Best learned gap vs persistence: {_fmt_metric(gap)}"
    if bool(holdout_coverage_summary.get("narrow_regime_support", False)):
        holdout_caption += " The current holdout covers a narrow operating regime."
    if bool(supplemental_surface_advisory.get("learned_beats_persistence", False)):
        regimes = ", ".join(supplemental_surface_advisory.get("learned_supported_operating_regimes", [])) or "unknown"
        holdout_caption += f" Broader advisory evidence is learned-positive in: {regimes}."
    holdout_caption += _supplemental_surface_high_signal_note(supplemental_surface_segment_evaluation)
    runtime_hotspot = str(runtime_summary.get("longest_step", "")).strip()
    runtime_hotspot_seconds = _safe_float(runtime_summary.get("longest_step_seconds"))
    validation_caption = "Latest modeling, rollout, control, and notebook manifests available to the dashboard."
    if runtime_hotspot:
        validation_caption += (
            f" Current Stage-10 hotspot: {runtime_hotspot} ({_fmt_metric(runtime_hotspot_seconds)}s)."
        )

    return [
        {
            "eyebrow": "1-minute decision",
            "value": str(deployment_recommendation.get("recommended_candidate_label", "n/a")),
            "caption": holdout_caption,
            "tone": "needs_work" if str(deployment_recommendation.get("recommended_candidate_label", "")) == "persistence" else "pass",
        },
        {
            "eyebrow": "Horizon wins",
            "value": f"{horizon_win_count}/{max(len(horizon), 1)}",
            "caption": "Current Stage-8 rows beating persistence on their selected objective.",
            "tone": "pass" if horizon_win_count > 0 else "mixed",
        },
        {
            "eyebrow": "Control-layer lift",
            "value": _fmt_pct(exact_reduction),
            "caption": f"Exact lock MAE reduction; rolling lock MAE reduction is {_fmt_pct(rolling_reduction)}.",
            "tone": "pass" if np.isfinite(exact_reduction) and exact_reduction > 50.0 else "mixed",
        },
        {
            "eyebrow": "Validation surface",
            "value": f"{success_count}/{len(manifest_rows)} green",
            "caption": validation_caption,
            "tone": "pass" if success_count == len(manifest_rows) else "mixed",
        },
    ]


def _rollout_winner_rows(rollout_summary: pd.DataFrame) -> list[dict[str, str]]:
    """Build a compact Stage-7 winner table."""
    if rollout_summary.empty:
        return []
    rows: list[dict[str, str]] = []
    for _, row in rollout_summary.iterrows():
        rows.append(
            {
                "selection_target": str(row.get("selection_target", "n/a")),
                "winner": str(row.get("winner_candidate_label", "n/a")),
                "metric_value": _fmt_metric(row.get("winner_metric_value")),
                "metric_pct": _fmt_pct(row.get("winner_metric_pct")),
                "support": f"{int(_safe_float(row.get('origin_n')))} origins" if np.isfinite(_safe_float(row.get("origin_n"))) else "n/a",
                "decision_reason": str(row.get("decision_reason", "n/a")),
            }
        )
    return rows


def _parse_figure_guide(guide_path: Path) -> tuple[str, list[dict[str, str]]]:
    """Parse a stage-local figure guide written by the modeling helpers."""
    try:
        lines = guide_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ("", [])

    stage_title = ""
    stage_purpose_lines: list[str] = []
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    header_pattern = re.compile(r"^### `([^`]+)`: (.+)$")
    for line in lines:
        if line.startswith("## ") and not stage_title:
            stage_title = line[3:].strip()
            continue
        if line.startswith("### "):
            match = header_pattern.match(line.strip())
            if match is None:
                continue
            if current is not None:
                entries.append(current)
            current = {
                "filename": match.group(1).strip(),
                "title": match.group(2).strip(),
                "intent": "",
                "how_to_read": "",
                "look_for": "",
            }
            continue
        if current is None:
            if stage_title and line.strip():
                stage_purpose_lines.append(line.strip())
            continue
        stripped = line.strip()
        if stripped.startswith("- Intent:"):
            current["intent"] = stripped.replace("- Intent:", "", 1).strip()
        elif stripped.startswith("- How to read it:"):
            current["how_to_read"] = stripped.replace("- How to read it:", "", 1).strip()
        elif stripped.startswith("- What to look for:"):
            current["look_for"] = stripped.replace("- What to look for:", "", 1).strip()
    if current is not None:
        entries.append(current)
    stage_purpose = " ".join(stage_purpose_lines).strip()
    return (stage_title, [{"stage_purpose": stage_purpose, **entry} for entry in entries])


def _artifact_gallery_sections(
    *,
    latest_paths: dict[str, Path],
    dashboard_path: Path,
) -> list[dict[str, Any]]:
    """Build the cross-stage artifact gallery metadata."""
    stage_roots = [
        ("Stage-4 notebook benchmark", latest_paths["modeling"]),
        ("Stage-5 holdout gate", latest_paths["performance"]),
        ("Stage-6 multiresolution", latest_paths["multires"]),
        ("Stage-7 rollout", latest_paths["rollout"]),
        ("Stage-8 horizon curve", latest_paths["horizon_curve"]),
        ("Stage-10 forecast control", latest_paths["forecast_control"]),
    ]
    sections: list[dict[str, Any]] = []
    for default_title, stage_root in stage_roots:
        guide_path = stage_root / "figure_guide.md"
        stage_title, entries = _parse_figure_guide(guide_path)
        resolved_title = stage_title or default_title
        stage_cards: list[dict[str, str]] = []
        for entry in entries:
            image_path = stage_root / entry["filename"]
            if not image_path.exists():
                continue
            stage_cards.append(
                {
                    "filename": entry["filename"],
                    "title": entry["title"],
                    "intent": entry["intent"],
                    "how_to_read": entry["how_to_read"],
                    "look_for": entry["look_for"],
                    "stage_purpose": entry["stage_purpose"],
                    "image_rel": _rel(image_path, dashboard_path.parent),
                    "guide_rel": _rel(guide_path, dashboard_path.parent),
                }
            )
        if stage_cards:
            sections.append(
                {
                    "title": resolved_title,
                    "guide_rel": _rel(guide_path, dashboard_path.parent),
                    "stage_purpose": stage_cards[0]["stage_purpose"],
                    "cards": stage_cards,
                }
            )
    return sections


def _recommended_markdown_embeds(sections: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Curate the figures that should be embedded inline in markdown surfaces."""
    card_lookup: dict[str, dict[str, str]] = {}
    for section in sections:
        for card in section["cards"]:
            card_lookup[card["filename"]] = {
                "stage": section["title"],
                **card,
            }

    recommendations = [
        (
            "fig_model_comparison.png",
            "Stage-4 benchmark overview",
            "Core inline",
            "Modeling notebook, milestone summaries, README-level overviews",
            "Which validation winners look strongest once baselines and coverage risk are visible in one chart?",
            "Use in modeling summaries and milestone markdowns when you need one benchmark-oriented view of the 1-minute surface.",
        ),
        (
            "fig_holdout_benchmark_ci.png",
            "Stage-5 honest 1-minute gate",
            "Core inline",
            "Validation snapshots, deployment notes, executive summaries",
            "Does any learned 1-minute challenger actually beat persistence on honest holdout with uncertainty shown?",
            "Use whenever the question is whether a learned 1-minute model really beats persistence on holdout.",
        ),
        (
            "fig_runtime_vs_gain.png",
            "Stage-6 compute-vs-value tradeoff",
            "Context inline",
            "Methodology markdowns, tradeoff sections, horizon-selection notes",
            "Which candidates buy enough MAE improvement to justify their runtime cost?",
            "Use when explaining why some learned candidates are interesting but still not worth their runtime cost.",
        ),
        (
            "fig_rollout_paths.png",
            "Stage-7 rollout behavior",
            "Context inline",
            "Rollout methodology docs, narrative result sections",
            "Which rollout candidate tracks ramps, plateaus, and phase timing with the least visible path drift?",
            "Use when you need one qualitative picture of how recursive rollout candidates track the actual path over time.",
        ),
        (
            "fig_horizon_ratio_curve.png",
            "Stage-8 horizon capability curve",
            "Core inline",
            "Validation snapshots, report abstracts, cross-stage summaries",
            "At which horizons does the learned stack beat persistence on the objective that actually matters there?",
            "Use as the default cross-horizon figure because it shows where learned models cross above or below persistence on the right objective.",
        ),
        (
            "fig_control_layer_gain_ci.png",
            "Stage-10 stacked control gains",
            "Core inline",
            "Operational summaries, control-stack sections, result highlights",
            "Do hourly, phase, and nowcast updates deliver statistically meaningful control-stack gains?",
            "Use when you need the strongest statistical evidence that hourly, phase, and nowcast updates really improve the stack.",
        ),
        (
            "fig_day_ahead_refresh_policy.png",
            "Stage-10 refresh policy choice",
            "Policy inline",
            "Optimizer-facing docs, control-policy notes, appendix markdowns",
            "Which refresh behavior gives the best policy tradeoff between improvement and update frequency?",
            "Use when documenting why the repo prefers frozen, unconditional, or triggered day-ahead refresh behavior.",
        ),
    ]

    rows: list[dict[str, str]] = []
    for filename, title, embed_tier, primary_markdowns, decision_question, why_embed in recommendations:
        card = card_lookup.get(filename)
        if card is None:
            continue
        rows.append(
            {
                "stage": str(card["stage"]),
                "filename": str(filename),
                "title": str(title),
                "embed_tier": str(embed_tier),
                "primary_markdowns": str(primary_markdowns),
                "decision_question": str(decision_question),
                "why_embed": str(why_embed),
                "image_rel": str(card["image_rel"]),
                "how_to_read": str(card["how_to_read"]),
                "look_for": str(card["look_for"]),
                "intent": str(card["intent"]),
            }
        )
    return rows


def _artifact_gallery_html(sections: list[dict[str, Any]]) -> str:
    """Render the cross-stage PNG gallery for the dashboard."""
    if not sections:
        return "<p class='empty-state'>No stage-local figure guides were available.</p>"
    section_html: list[str] = []
    for section in sections:
        cards_html: list[str] = []
        for card in section["cards"]:
            cards_html.append(
                "<article class='gallery-card'>"
                f"<img src='{html.escape(card['image_rel'])}' alt='{html.escape(card['title'])}' loading='lazy' />"
                "<div class='gallery-copy'>"
                f"<h4>{html.escape(card['filename'])}</h4>"
                f"<p class='gallery-title'>{html.escape(card['title'])}</p>"
                f"<p><strong>Intent:</strong> {html.escape(card['intent'])}</p>"
                f"<p><strong>How to read:</strong> {html.escape(card['how_to_read'])}</p>"
                f"<p><strong>Look for:</strong> {html.escape(card['look_for'])}</p>"
                "</div>"
                "</article>"
            )
        section_html.append(
            "<details class='gallery-section'>"
            f"<summary>{html.escape(section['title'])} "
            f"<span class='gallery-link'><a href='{html.escape(section['guide_rel'])}'>open figure guide</a></span></summary>"
            f"<p class='gallery-stage-purpose'>{html.escape(section['stage_purpose'])}</p>"
            f"<div class='gallery-grid'>{''.join(cards_html)}</div>"
            "</details>"
        )
    return "".join(section_html)


def _artifact_gallery_markdown(
    *,
    sections: list[dict[str, Any]],
    doc_dir: Path,
    dashboard_path: Path,
) -> str:
    """Render the stage-local artifact gallery for the markdown guide."""
    if not sections:
        return "_No stage-local figure guides were available._"
    lines: list[str] = []
    for section in sections:
        lines.extend(
            [
                f"### {section['title']}",
                "",
                section["stage_purpose"],
                "",
                f"- Figure guide: [{_rel((dashboard_path.parent / section['guide_rel']).resolve(), doc_dir)}]({_rel((dashboard_path.parent / section['guide_rel']).resolve(), doc_dir)})",
            ]
        )
        for card in section["cards"]:
            image_abs = (dashboard_path.parent / card["image_rel"]).resolve()
            lines.extend(
                [
                    f"- `{card['filename']}`: {card['title']}",
                    f"  - Intent: {card['intent']}",
                    f"  - How to read it: {card['how_to_read']}",
                    f"  - What to look for: {card['look_for']}",
                    f"  - Image: [{_rel(image_abs, doc_dir)}]({_rel(image_abs, doc_dir)})",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _figure_html(fig: go.Figure | None, *, include_plotlyjs: bool) -> str:
    """Render one plotly figure into an embeddable HTML fragment."""
    if fig is None:
        return "<div class='empty-chart'>Latest artifacts were not sufficient to build this visual.</div>"
    return pio.to_html(
        fig,
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        config={"displayModeBar": False, "responsive": True},
    )


def _goal_cards_html(goal_rows: list[dict[str, str]]) -> str:
    """Render the goal status cards for the dashboard header."""
    blocks: list[str] = []
    for row in goal_rows:
        blocks.append(
            f"<article class='goal-card {_STATUS_CLASS.get(row['status'], 'status-mixed')}'>"
            f"<div class='goal-meta'><span class='goal-label'>{html.escape(row['goal'])}</span>"
            f"<span class='goal-status'>{html.escape(row['status_label'])}</span></div>"
            f"<p class='goal-readout'>{html.escape(row['current_readout'])}</p>"
            f"<p class='goal-metric'><strong>Primary metric:</strong> {html.escape(row['primary_metric'])}</p>"
            f"<p class='goal-success'><strong>Success means:</strong> {html.escape(row['success_looks_like'])}</p>"
            "</article>"
        )
    return "".join(blocks)


def _summary_cards_html(cards: list[dict[str, str]]) -> str:
    """Render the hero summary cards for the dashboard."""
    blocks: list[str] = []
    for card in cards:
        blocks.append(
            f"<article class='summary-card {_STATUS_CLASS.get(card['tone'], 'status-mixed')}'>"
            f"<p class='summary-eyebrow'>{html.escape(card['eyebrow'])}</p>"
            f"<p class='summary-value'>{html.escape(card['value'])}</p>"
            f"<p class='summary-caption'>{html.escape(card['caption'])}</p>"
            "</article>"
        )
    return "".join(blocks)


def build_visualization_report_markdown(
    *,
    artifact_namespace: str,
    generated_at_utc: str,
    goal_rows: list[dict[str, str]],
    summary_cards: list[dict[str, str]],
    stage_question_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
    rollout_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    integrated_visuals: list[dict[str, str]],
    markdown_embed_rows: list[dict[str, str]],
    artifact_sections: list[dict[str, Any]],
    doc_output_path: Path,
    dashboard_output_path: Path,
) -> str:
    """Build the markdown companion guide for the integrated dashboard."""
    lines = [
        "# Current Visualization Guide",
        "",
        "This guide is generated from the latest persisted artifacts. It exists to",
        "answer three recurring questions quickly:",
        "",
        "- What are the repo's current goals and success criteria?",
        "- Which metrics are authoritative for each decision layer?",
        "- What does each visual answer, and what should you look for?",
        "",
        f"- Generated at: `{generated_at_utc}`",
        f"- Artifact namespace: `{artifact_namespace}`",
        f"- Integrated dashboard: [{_rel(dashboard_output_path, doc_output_path.parent)}]({_rel(dashboard_output_path, doc_output_path.parent)})",
        "",
        "## Executive Summary",
        "",
    ]
    for card in summary_cards:
        lines.append(f"- {card['eyebrow']}: `{card['value']}`. {card['caption']}")
    lines.extend(
        [
            "",
            "## Goals",
            "",
            _table_markdown(
                goal_rows,
                [
                    ("goal", "Goal"),
                    ("decision_layer", "Decision layer"),
                    ("primary_metric", "Primary metric"),
                    ("current_readout", "Current readout"),
                    ("status_label", "Status"),
                ],
            ),
            "",
            "## Decision Layers",
            "",
            _table_markdown(
                stage_question_rows,
                [
                    ("stage", "Stage"),
                    ("question", "Decision question"),
                    ("authoritative_metric", "Authoritative metric"),
                    ("main_visual", "Main visual"),
                ],
            ),
            "",
            "## Success Metrics",
            "",
            _table_markdown(
                metric_rows,
                [
                    ("metric", "Metric"),
                    ("use_case", "Use this when"),
                    ("authoritative_scope", "Authoritative scope"),
                    ("good_pattern", "A good pattern looks like"),
                    ("warning", "Do not over-interpret"),
                ],
            ),
            "",
            "## Primary Markdown Embeds",
            "",
            "Treat `Core inline` visuals as the default embeds for notebook narrative cells, canonical snapshots, and report summaries. Use `Context inline` visuals only when the section is specifically about compute tradeoffs or rollout realism, and reserve `Policy inline` visuals for control-policy and optimizer-facing markdowns.",
            "",
            _table_markdown(
                markdown_embed_rows,
                [
                    ("stage", "Stage"),
                    ("filename", "Visual"),
                    ("embed_tier", "Embed tier"),
                    ("primary_markdowns", "Primary markdown homes"),
                    ("decision_question", "Decision question"),
                ],
            ),
            "",
            "## Embedded Recommended Visuals",
            "",
        ]
    )
    if not markdown_embed_rows:
        lines.extend(["_No embed-ready stage figures were available from the latest artifacts._", ""])
    for row in markdown_embed_rows:
        image_abs = (dashboard_output_path.parent / row["image_rel"]).resolve()
        lines.extend(
            [
                f"### {row['stage']}: {row['title']}",
                "",
                f"- Why embed it: {row['why_embed']}",
                f"- How to read it: {row['how_to_read']}",
                f"- What to look for: {row['look_for']}",
                "",
                f"![{row['title']}]({_rel(image_abs, doc_output_path.parent)})",
                "",
            ]
        )
    lines.extend(
        [
            "## Integrated Visuals",
            "",
        ]
    )
    for visual in integrated_visuals:
        lines.extend(
            [
                f"### {visual['title']}",
                "",
                f"- Purpose: {visual['purpose']}",
                f"- How to read it: {visual['how_to_read']}",
                f"- What to look for: {visual['look_for']}",
                f"- Source artifacts: {visual['sources']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Stage-7 Objective Winners",
            "",
            _table_markdown(
                rollout_rows,
                [
                    ("selection_target", "Selection target"),
                    ("winner", "Winner"),
                    ("metric_value", "Metric value"),
                    ("metric_pct", "Metric %"),
                    ("support", "Support"),
                    ("decision_reason", "Decision reason"),
                ],
            ),
            "",
            "## Manifest Health",
            "",
            _table_markdown(
                manifest_rows,
                [("stage", "Surface"), ("status", "Status"), ("run_id", "Run id")],
            ),
            "",
            "## Existing Artifact Gallery",
            "",
            _artifact_gallery_markdown(
                sections=artifact_sections,
                doc_dir=doc_output_path.parent,
                dashboard_path=dashboard_output_path,
            ),
            "",
            "## Supporting References",
            "",
            "- [current_validation_snapshot.md](current_validation_snapshot.md)",
            "- [current_operating_approach.md](current_operating_approach.md)",
            "- [model_and_blend_guide.md](model_and_blend_guide.md)",
            "- [hypothesis.md](hypothesis.md)",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_visualization_dashboard_html(
    *,
    artifact_namespace: str,
    generated_at_utc: str,
    summary_cards: list[dict[str, str]],
    goal_rows: list[dict[str, str]],
    stage_question_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
    rollout_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    integrated_visuals: list[dict[str, str]],
    markdown_embed_rows: list[dict[str, str]],
    chart_html_map: dict[str, str],
    artifact_sections: list[dict[str, Any]],
) -> str:
    """Build the standalone HTML dashboard."""
    goal_table = _table_html(
        goal_rows,
        [
            ("goal", "Goal"),
            ("decision_layer", "Decision layer"),
            ("primary_metric", "Primary metric"),
            ("current_readout", "Current readout"),
            ("status_label", "Status"),
        ],
    )
    stage_table = _table_html(
        stage_question_rows,
        [
            ("stage", "Stage"),
            ("question", "Decision question"),
            ("authoritative_metric", "Authoritative metric"),
            ("main_visual", "Main visual"),
        ],
    )
    metric_table = _table_html(
        metric_rows,
        [
            ("metric", "Metric"),
            ("use_case", "Use this when"),
            ("authoritative_scope", "Authoritative scope"),
            ("good_pattern", "A good pattern looks like"),
            ("warning", "Do not over-interpret"),
        ],
    )
    rollout_table = _table_html(
        rollout_rows,
        [
            ("selection_target", "Selection target"),
            ("winner", "Winner"),
            ("metric_value", "Metric value"),
            ("metric_pct", "Metric %"),
            ("support", "Support"),
            ("decision_reason", "Decision reason"),
        ],
    )
    markdown_embed_table = _table_html(
        markdown_embed_rows,
        [
            ("stage", "Stage"),
            ("filename", "Visual"),
            ("embed_tier", "Embed tier"),
            ("primary_markdowns", "Primary markdown homes"),
            ("decision_question", "Decision question"),
        ],
    )
    manifest_table = _table_html(
        manifest_rows,
        [("stage", "Surface"), ("status", "Status"), ("run_id", "Run id")],
    )
    visuals_html: list[str] = []
    for visual in integrated_visuals:
        visuals_html.append(
            "<section class='visual-card'>"
            "<div class='visual-copy'>"
            f"<p class='visual-eyebrow'>{html.escape(visual['sources'])}</p>"
            f"<h3>{html.escape(visual['title'])}</h3>"
            f"<p><strong>Purpose:</strong> {html.escape(visual['purpose'])}</p>"
            f"<p><strong>How to read:</strong> {html.escape(visual['how_to_read'])}</p>"
            f"<p><strong>What to look for:</strong> {html.escape(visual['look_for'])}</p>"
            "</div>"
            f"<div class='visual-chart'>{chart_html_map.get(visual['id'], '<div class=\"empty-chart\">This visual could not be built from the latest artifacts.</div>')}</div>"
            "</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Electric Load Forecasting Validation Dashboard</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --ink: #0f172a;
      --muted: #475569;
      --card: rgba(255, 255, 255, 0.9);
      --line: rgba(148, 163, 184, 0.32);
      --teal: #0ea5e9;
      --green: #16a34a;
      --amber: #d97706;
      --red: #dc2626;
      --shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Aptos, "Segoe UI Variable", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(14, 165, 233, 0.12), transparent 34%),
        radial-gradient(circle at top right, rgba(249, 115, 22, 0.12), transparent 30%),
        linear-gradient(180deg, #fbfdff 0%, var(--bg) 48%, #eef3f8 100%);
    }}
    .shell {{ width: min(1380px, calc(100vw - 32px)); margin: 0 auto; padding: 28px 0 56px; }}
    .hero {{
      padding: 28px;
      border-radius: 28px;
      background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(17, 94, 89, 0.92));
      color: #eff6ff;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{ margin: 0 0 12px; font-size: clamp(2rem, 3.6vw, 3.4rem); line-height: 1.03; letter-spacing: -0.04em; }}
    .hero p {{ margin: 0; max-width: 960px; color: rgba(239, 246, 255, 0.9); font-size: 1rem; line-height: 1.6; }}
    .hero-meta {{ display: flex; flex-wrap: wrap; gap: 10px 16px; margin-top: 18px; color: rgba(239, 246, 255, 0.82); font-size: 0.95rem; }}
    .summary-grid, .goal-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; margin-top: 18px; }}
    .summary-card, .goal-card {{ border: 1px solid var(--line); background: var(--card); box-shadow: var(--shadow); border-radius: 22px; padding: 18px 18px 16px; }}
    .summary-eyebrow, .visual-eyebrow, .goal-label, .section-eyebrow {{
      margin: 0 0 10px; text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.72rem; color: var(--muted);
    }}
    .summary-value {{ margin: 0 0 6px; font-size: 1.9rem; font-weight: 700; letter-spacing: -0.04em; }}
    .summary-caption, .goal-readout, .goal-metric, .goal-success {{ margin: 0; color: var(--muted); line-height: 1.55; }}
    .goal-meta {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }}
    .goal-status {{ font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }}
    .status-pass {{ border-color: rgba(22, 163, 74, 0.24); }}
    .status-pass .goal-status, .status-pass .summary-value {{ color: var(--green); }}
    .status-mixed {{ border-color: rgba(217, 119, 6, 0.24); }}
    .status-mixed .goal-status, .status-mixed .summary-value {{ color: var(--amber); }}
    .status-needs-work {{ border-color: rgba(220, 38, 38, 0.24); }}
    .status-needs-work .goal-status, .status-needs-work .summary-value {{ color: var(--red); }}
    .section {{
      margin-top: 22px; padding: 24px; border-radius: 26px; background: rgba(255, 255, 255, 0.72);
      border: 1px solid rgba(226, 232, 240, 0.78); box-shadow: var(--shadow);
    }}
    .section h2 {{ margin: 0 0 8px; font-size: clamp(1.4rem, 2vw, 2rem); letter-spacing: -0.03em; }}
    .section p.section-copy {{ margin: 0 0 16px; color: var(--muted); line-height: 1.6; max-width: 980px; }}
    .table-shell {{ overflow-x: auto; }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.95rem; background: rgba(255, 255, 255, 0.78); border-radius: 18px; overflow: hidden; }}
    .data-table th, .data-table td {{ text-align: left; padding: 12px 14px; border-bottom: 1px solid rgba(226, 232, 240, 0.82); vertical-align: top; line-height: 1.45; }}
    .data-table th {{ background: rgba(226, 232, 240, 0.36); color: var(--ink); font-size: 0.84rem; text-transform: uppercase; letter-spacing: 0.08em; }}
    .visual-stack {{ display: grid; gap: 18px; }}
    .visual-card {{
      display: grid; grid-template-columns: minmax(300px, 420px) minmax(0, 1fr); gap: 18px; padding: 18px;
      background: rgba(255, 255, 255, 0.82); border: 1px solid rgba(226, 232, 240, 0.8); border-radius: 24px; box-shadow: var(--shadow);
    }}
    .visual-copy h3 {{ margin: 0 0 10px; font-size: 1.35rem; letter-spacing: -0.03em; }}
    .visual-copy p {{ margin: 0 0 10px; color: var(--muted); line-height: 1.6; }}
    .visual-chart {{ min-height: 320px; }}
    .empty-chart, .empty-state {{ padding: 16px; border-radius: 18px; background: rgba(241, 245, 249, 0.84); color: var(--muted); }}
    .gallery-section {{ margin-top: 16px; border: 1px solid rgba(226, 232, 240, 0.78); border-radius: 20px; background: rgba(255, 255, 255, 0.78); overflow: hidden; }}
    .gallery-section summary {{ list-style: none; cursor: pointer; padding: 18px 20px; font-size: 1.02rem; font-weight: 700; display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
    .gallery-link a {{ color: var(--teal); text-decoration: none; font-weight: 600; font-size: 0.9rem; }}
    .gallery-stage-purpose {{ margin: 0; padding: 0 20px 12px; color: var(--muted); line-height: 1.6; }}
    .gallery-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; padding: 0 20px 20px; }}
    .gallery-card {{ border: 1px solid rgba(226, 232, 240, 0.86); border-radius: 18px; overflow: hidden; background: rgba(255, 255, 255, 0.9); }}
    .gallery-card img {{ width: 100%; height: 220px; object-fit: cover; display: block; background: #e2e8f0; }}
    .gallery-copy {{ padding: 14px; }}
    .gallery-copy h4, .gallery-copy p {{ margin: 0 0 10px; line-height: 1.55; }}
    .gallery-title {{ font-weight: 700; color: var(--ink); }}
    @media (max-width: 920px) {{
      .shell {{ width: min(100vw - 20px, 1380px); padding-top: 12px; }}
      .hero, .section {{ border-radius: 22px; padding: 20px; }}
      .visual-card {{ grid-template-columns: 1fr; }}
      .gallery-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <p class="section-eyebrow">Integrated validation dashboard</p>
      <h1>Electric Load Forecasting Visual Story</h1>
      <p>
        This dashboard is artifact-driven. It does not replace the stage-specific reports; it integrates
        them so the latest goals, success criteria, and decision-facing visuals can be read as one coherent operating story.
      </p>
      <div class="hero-meta">
        <span>Artifact namespace: <strong>{html.escape(artifact_namespace)}</strong></span>
        <span>Generated at: <strong>{html.escape(generated_at_utc)}</strong></span>
      </div>
      <div class="summary-grid">{_summary_cards_html(summary_cards)}</div>
    </section>

    <section class="section">
      <p class="section-eyebrow">Goals</p>
      <h2>What Success Means Right Now</h2>
      <p class="section-copy">Each goal is tied to the stage that is allowed to answer it, so the dashboard does not let one good chart overrule the wrong decision layer.</p>
      <div class="goal-grid">{_goal_cards_html(goal_rows)}</div>
    </section>

    <section class="section">
      <p class="section-eyebrow">Decision map</p>
      <h2>Authoritative Decision Layers</h2>
      <p class="section-copy">Different stages answer different questions. The repo is strongest when each visual is read in the decision layer it was designed for.</p>
      {stage_table}
    </section>

    <section class="section">
      <p class="section-eyebrow">Metrics</p>
      <h2>Success Metrics And How To Use Them</h2>
      <p class="section-copy">The same forecast can look strong or weak depending on which operational question it is being asked to answer. This glossary makes the dashboard's scorekeeping explicit.</p>
      {metric_table}
    </section>

    <section class="section">
      <p class="section-eyebrow">Integrated visuals</p>
      <h2>Cross-Stage Diagnostics</h2>
      <p class="section-copy">Each visual below exists to answer one decision-facing question. The guidance beside each chart explains what it is for, how to read it, and what pattern should count as a win or a warning.</p>
      <div class="visual-stack">{''.join(visuals_html)}</div>
    </section>

    <section class="section">
      <p class="section-eyebrow">Markdown curation</p>
      <h2>Which Visuals Belong Inline In Markdown</h2>
      <p class="section-copy">These recommendations separate core inline visuals from context and policy visuals, and name the markdown surfaces where each one earns the space.</p>
      {markdown_embed_table}
    </section>

    <section class="section">
      <p class="section-eyebrow">Rollout objective table</p>
      <h2>Stage-7 Objective Winners</h2>
      <p class="section-copy">Rollout policy selection is objective-aware. These rows explain which candidate currently wins once the objective changes from endpoint accuracy to path, phase, next-lock, or day-profile quality.</p>
      {rollout_table}
    </section>

    <section class="section">
      <p class="section-eyebrow">Manifest health</p>
      <h2>Latest Evidence Chain</h2>
      <p class="section-copy">The reporting surface is only trustworthy when the artifact chain behind it is current and green.</p>
      {manifest_table}
    </section>

    <section class="section">
      <p class="section-eyebrow">Existing figures</p>
      <h2>Stage-Local Artifact Gallery</h2>
      <p class="section-copy">The interactive dashboard is additive, not a replacement for the existing PNG surfaces. These sections pull the latest stage-local figures into one place and carry forward their original reading guidance.</p>
      {_artifact_gallery_html(artifact_sections)}
    </section>
  </main>
</body>
</html>
"""


def write_visualization_report(
    *,
    project_root: Path = PROJECT_ROOT,
    artifact_namespace: str = DATASET["artifact_namespace"],
    doc_output_path: Path = DEFAULT_DOC_OUTPUT_PATH,
    dashboard_output_path: Path = DEFAULT_HTML_OUTPUT_PATH,
    generated_at_utc: str | None = None,
    embed_plotlyjs: bool = True,
) -> VisualizationReportPaths:
    """Write the integrated markdown guide and HTML dashboard."""
    project_root = Path(project_root).resolve()
    doc_output_path = Path(doc_output_path).resolve()
    dashboard_output_path = Path(dashboard_output_path).resolve()
    generated_at_utc = generated_at_utc or datetime.now(UTC).isoformat()

    latest_paths = _latest_paths(project_root, str(artifact_namespace))
    manifests = {
        stage_key: _read_json(stage_root / "run_manifest.json") or {}
        for stage_key, stage_root in latest_paths.items()
        if stage_key != "modeling"
    }
    manifests["modeling"] = _read_json(latest_paths["modeling"] / "run_manifest.json") or {}

    holdout = _read_csv(latest_paths["performance"] / "holdout_evaluation.csv")
    holdout_coverage_summary = _read_json(latest_paths["performance"] / "holdout_coverage_summary.json") or {}
    supplemental_surface_advisory = (
        _read_json(latest_paths["performance"] / "supplemental_surface_advisory.json") or {}
    )
    supplemental_surface_segment_evaluation = _read_csv(
        latest_paths["performance"] / "supplemental_surface_segment_evaluation.csv"
    )
    multires_metrics = _read_csv(latest_paths["multires"] / "matched_horizon_metrics.csv")
    rollout_summary = _read_csv(latest_paths["rollout"] / "rollout_selection_summary.csv")
    horizon = _read_csv(latest_paths["horizon_curve"] / "horizon_curve_summary.csv")
    control_summary = _read_csv(latest_paths["forecast_control"] / "control_backtest_summary.csv")
    rolling_summary = _read_csv(latest_paths["forecast_control"] / "rolling_control_backtest_summary.csv")
    exact_by_cycle = _read_csv(latest_paths["forecast_control"] / "control_backtest_by_cycle.csv")
    rolling_by_cycle = _read_csv(latest_paths["forecast_control"] / "rolling_control_backtest_by_cycle.csv")
    refresh_summary = _read_csv(latest_paths["forecast_control"] / "day_ahead_refresh_summary.csv")
    rolling_inference = _read_csv(latest_paths["forecast_control"] / "rolling_control_layer_inference.csv")
    runtime_summary = _read_json(latest_paths["forecast_control"] / "runtime_summary.json") or {}
    deployment_recommendation = _read_json(latest_paths["performance"] / "deployment_recommendation.json") or {}

    goal_rows = _goal_rows(
        manifests=manifests,
        holdout=holdout,
        deployment_recommendation=deployment_recommendation,
        horizon=horizon,
        control_summary=control_summary,
        rolling_summary=rolling_summary,
        rolling_inference=rolling_inference,
    )
    summary_cards = _summary_cards(
        manifests=manifests,
        deployment_recommendation=deployment_recommendation,
        holdout=holdout,
        holdout_coverage_summary=holdout_coverage_summary,
        supplemental_surface_advisory=supplemental_surface_advisory,
        supplemental_surface_segment_evaluation=supplemental_surface_segment_evaluation,
        horizon=horizon,
        control_summary=control_summary,
        rolling_summary=rolling_summary,
        runtime_summary=runtime_summary,
    )
    stage_question_rows = _stage_question_rows()
    metric_rows = _metric_catalog_rows()
    rollout_rows = _rollout_winner_rows(rollout_summary)
    manifest_rows = _manifest_health_rows(manifests)
    integrated_visuals = _integrated_visual_specs()
    artifact_sections = _artifact_gallery_sections(
        latest_paths=latest_paths,
        dashboard_path=dashboard_output_path,
    )
    markdown_embed_rows = _recommended_markdown_embeds(artifact_sections)

    figures = {
        "holdout_leaderboard": _build_holdout_figure(holdout),
        "multires_tradeoff": _build_multires_figure(multires_metrics),
        "horizon_ratio_curve": _build_horizon_ratio_figure(horizon),
        "horizon_win_matrix": _build_horizon_matrix_figure(horizon),
        "control_layer_trajectory": _build_control_layer_figure(control_summary, rolling_summary),
        "control_cycle_distribution": _build_cycle_distribution_figure(exact_by_cycle, rolling_by_cycle),
        "refresh_policy_comparison": _build_refresh_figure(refresh_summary),
    }
    chart_html_map: dict[str, str] = {}
    include_plotlyjs = bool(embed_plotlyjs)
    for visual in integrated_visuals:
        chart_html_map[visual["id"]] = _figure_html(
            figures.get(visual["id"]),
            include_plotlyjs=include_plotlyjs,
        )
        include_plotlyjs = False

    markdown_content = build_visualization_report_markdown(
        artifact_namespace=str(artifact_namespace),
        generated_at_utc=generated_at_utc,
        goal_rows=goal_rows,
        summary_cards=summary_cards,
        stage_question_rows=stage_question_rows,
        metric_rows=metric_rows,
        rollout_rows=rollout_rows,
        manifest_rows=manifest_rows,
        integrated_visuals=integrated_visuals,
        markdown_embed_rows=markdown_embed_rows,
        artifact_sections=artifact_sections,
        doc_output_path=doc_output_path,
        dashboard_output_path=dashboard_output_path,
    )
    html_content = build_visualization_dashboard_html(
        artifact_namespace=str(artifact_namespace),
        generated_at_utc=generated_at_utc,
        summary_cards=summary_cards,
        goal_rows=goal_rows,
        stage_question_rows=stage_question_rows,
        metric_rows=metric_rows,
        rollout_rows=rollout_rows,
        manifest_rows=manifest_rows,
        integrated_visuals=integrated_visuals,
        markdown_embed_rows=markdown_embed_rows,
        chart_html_map=chart_html_map,
        artifact_sections=artifact_sections,
    )

    doc_output_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_output_path.parent.mkdir(parents=True, exist_ok=True)
    doc_output_path.write_text(markdown_content, encoding="utf-8")
    dashboard_output_path.write_text(html_content, encoding="utf-8")
    return VisualizationReportPaths(
        guide_path=doc_output_path,
        dashboard_path=dashboard_output_path,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for standalone visualization-report generation."""
    parser = argparse.ArgumentParser(description="Write the integrated visualization guide and dashboard.")
    parser.add_argument("--artifact-namespace", default=DATASET["artifact_namespace"])
    parser.add_argument("--doc-output-path", default=str(DEFAULT_DOC_OUTPUT_PATH))
    parser.add_argument("--dashboard-output-path", default=str(DEFAULT_HTML_OUTPUT_PATH))
    parser.add_argument(
        "--no-embed-plotlyjs",
        action="store_true",
        help="Do not inline the plotly runtime into the HTML dashboard.",
    )
    return parser.parse_args()


def main() -> int:
    """Write the latest integrated visualization report to disk."""
    args = parse_args()
    result = write_visualization_report(
        artifact_namespace=str(args.artifact_namespace),
        doc_output_path=Path(args.doc_output_path),
        dashboard_output_path=Path(args.dashboard_output_path),
        embed_plotlyjs=not bool(args.no_embed_plotlyjs),
    )
    print(f"[write_visualization_report] wrote guide {result.guide_path}")
    print(f"[write_visualization_report] wrote dashboard {result.dashboard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
