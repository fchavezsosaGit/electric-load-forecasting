"""Stage-8 H5 horizon-characterization curve across nowcast and rollout horizons.

This stage consolidates evidence from Stage-5 and Stage-7 into one objective-aware
capability envelope. It exists to answer the recurring question "where does the
current learned stack beat persistence or the best baseline, and on which metric?"
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

import sys

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
matplotlib.use("Agg")
import matplotlib.pyplot as plt
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import (  # noqa: E402
    DATASET,
    MULTIRES_HORIZON_CURVE,
    MULTIRES_ROLLOUT,
    PATHS,
    preferred_output_path,
    resolve_rollout_origin_policy,
    resolve_rollout_selection_target,
    scoped_output_path,
    validate_config,
)
from modeling.common import (  # noqa: E402
    FigureGuideEntry,
    stable_config_hash,
    update_latest_alias,
    validate_png_artifact,
    write_figure_guide,
)
from modeling.rollout_challenger_sweep import (  # noqa: E402
    _build_challenger_sweep_registry_snapshot,
    _select_challenger_sweep_registry_candidate,
    run_rollout_challenger_sweep,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = SCRIPTS_DIR.parent


def _configure_logging() -> None:
    """Initialize a basic logger for direct CLI execution of the stage."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _relative_artifact_path(path: Path) -> str:
    """Render an artifact path relative to the project root when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _project_artifact_path(path_value: str) -> Path:
    """Resolve a stored artifact reference into an absolute path."""
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a stable comparison ratio or `nan` for invalid denominators."""
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return float(numerator) / float(denominator)


def _load_stage5_nowcast_anchor() -> dict[str, Any]:
    """Load the current single-step learned-vs-persistence anchor from Stage-5 holdout."""
    performance_root = preferred_output_path(PATHS["outputs_performance_dir"])
    holdout_path = performance_root / "latest" / "holdout_evaluation.csv"
    if not holdout_path.exists():
        raise FileNotFoundError(f"Missing Stage-5 holdout evaluation artifact: {holdout_path}")

    holdout = pd.read_csv(holdout_path)
    learned = holdout.loc[holdout["candidate_type"].astype("string").ne("baseline")].copy()
    if learned.empty:
        raise RuntimeError(f"No learned holdout candidate found in {holdout_path}")
    learned = learned.sort_values(["mae", "candidate_label"], ascending=[True, True], kind="stable")
    learned_row = learned.iloc[0]

    persistence = holdout.loc[holdout["candidate_label"].astype("string").eq("persistence")].copy()
    if persistence.empty:
        raise RuntimeError(f"Missing persistence row in {holdout_path}")
    persistence_row = persistence.iloc[0]

    baselines = holdout.loc[holdout["candidate_type"].astype("string").eq("baseline")].copy()
    baselines = baselines.sort_values(["mae", "candidate_label"], ascending=[True, True], kind="stable")
    best_baseline = baselines.iloc[0]

    learned_mae = float(learned_row["mae"])
    learned_mae_pct = float(learned_row.get("mae_pct", float("nan")))
    persistence_mae = float(persistence_row["mae"])
    persistence_mae_pct = float(persistence_row.get("mae_pct", float("nan")))
    best_baseline_mae = float(best_baseline["mae"])
    best_baseline_mae_pct = float(best_baseline.get("mae_pct", float("nan")))
    return {
        "horizon_minutes": 1,
        "source_stage": "005_performance",
        "selection_policy": "stage5_holdout_anchor",
        "selection_target": "endpoint_mae",
        "candidate_label": str(learned_row["candidate_label"]),
        "resolution": str(learned_row["resolution"]),
        "feature_set": str(learned_row["feature_set"]),
        "model_label": str(learned_row["model_label"]),
        "selection_source": _relative_artifact_path(holdout_path),
        "learned_endpoint_mae": learned_mae,
        "learned_endpoint_mae_pct": learned_mae_pct,
        "learned_path_mae": learned_mae,
        "learned_path_mae_pct": learned_mae_pct,
        "learned_phase_mean_mae": learned_mae,
        "learned_phase_mean_mae_pct": learned_mae_pct,
        "learned_next_lock_mae": learned_mae,
        "learned_next_lock_mae_pct": learned_mae_pct,
        "learned_profile_shape_mae": learned_mae,
        "learned_profile_shape_mae_pct": learned_mae_pct,
        "learned_energy_mae": learned_mae,
        "learned_energy_mae_pct": learned_mae_pct,
        "persistence_endpoint_mae": persistence_mae,
        "persistence_endpoint_mae_pct": persistence_mae_pct,
        "persistence_path_mae": persistence_mae,
        "persistence_path_mae_pct": persistence_mae_pct,
        "persistence_phase_mean_mae": persistence_mae,
        "persistence_phase_mean_mae_pct": persistence_mae_pct,
        "persistence_next_lock_mae": persistence_mae,
        "persistence_next_lock_mae_pct": persistence_mae_pct,
        "persistence_profile_shape_mae": persistence_mae,
        "persistence_profile_shape_mae_pct": persistence_mae_pct,
        "best_baseline_endpoint_label": str(best_baseline["candidate_label"]),
        "best_baseline_endpoint_mae": best_baseline_mae,
        "best_baseline_endpoint_mae_pct": best_baseline_mae_pct,
        "best_baseline_path_label": str(best_baseline["candidate_label"]),
        "best_baseline_path_mae": best_baseline_mae,
        "best_baseline_path_mae_pct": best_baseline_mae_pct,
        "best_baseline_phase_label": str(best_baseline["candidate_label"]),
        "best_baseline_phase_mae": best_baseline_mae,
        "best_baseline_phase_mae_pct": best_baseline_mae_pct,
        "best_baseline_next_lock_label": str(best_baseline["candidate_label"]),
        "best_baseline_next_lock_mae": best_baseline_mae,
        "best_baseline_next_lock_mae_pct": best_baseline_mae_pct,
        "best_baseline_profile_shape_label": str(best_baseline["candidate_label"]),
        "best_baseline_profile_shape_mae": best_baseline_mae,
        "best_baseline_profile_shape_mae_pct": best_baseline_mae_pct,
        "endpoint_ratio_to_persistence": _safe_ratio(learned_mae, persistence_mae),
        "path_ratio_to_persistence": _safe_ratio(learned_mae, persistence_mae),
        "phase_ratio_to_persistence": _safe_ratio(learned_mae, persistence_mae),
        "next_lock_ratio_to_persistence": _safe_ratio(learned_mae, persistence_mae),
        "profile_shape_ratio_to_persistence": _safe_ratio(learned_mae, persistence_mae),
        "endpoint_ratio_to_best_baseline": _safe_ratio(learned_mae, best_baseline_mae),
        "path_ratio_to_best_baseline": _safe_ratio(learned_mae, best_baseline_mae),
        "phase_ratio_to_best_baseline": _safe_ratio(learned_mae, best_baseline_mae),
        "next_lock_ratio_to_best_baseline": _safe_ratio(learned_mae, best_baseline_mae),
        "profile_shape_ratio_to_best_baseline": _safe_ratio(learned_mae, best_baseline_mae),
        "beats_persistence_endpoint": learned_mae < persistence_mae,
        "beats_persistence_path": learned_mae < persistence_mae,
        "beats_persistence_phase": learned_mae < persistence_mae,
        "beats_persistence_next_lock": learned_mae < persistence_mae,
        "beats_persistence_profile_shape": learned_mae < persistence_mae,
        "beats_best_baseline_endpoint": learned_mae < best_baseline_mae,
        "beats_best_baseline_path": learned_mae < best_baseline_mae,
        "beats_best_baseline_phase": learned_mae < best_baseline_mae,
        "beats_best_baseline_next_lock": learned_mae < best_baseline_mae,
        "beats_best_baseline_profile_shape": learned_mae < best_baseline_mae,
        "support_count": int(learned_row["n_eval"]),
        "support_type": "rows",
        "requested_origin_policy": "n/a",
    }


def _curve_row_from_sweep(
    *,
    horizon_minutes: int,
    selection_target: str,
    origin_policy: str,
    sweep_result: dict[str, Any],
) -> dict[str, Any]:
    """Normalize one challenger-sweep recommendation into the H5 curve schema."""
    recommended = sweep_result["recommended"]
    return {
        "horizon_minutes": int(horizon_minutes),
        "source_stage": "007_rollout_challenger_sweep",
        "selection_policy": "rollout_challenger_sweep",
        "selection_target": str(selection_target),
        "recommended_origin_policy": str(recommended.get("recommended_origin_policy", origin_policy)),
        "candidate_label": str(recommended["candidate_label"]),
        "resolution": str(recommended["resolution"]),
        "feature_set": str(recommended["feature_set"]),
        "model_label": str(recommended["model_label"]),
        "selection_source": _relative_artifact_path(
            Path(sweep_result["sweep_dir"]) / "recommended_candidate.json"
        ),
        "learned_endpoint_mae": float(recommended["endpoint_mae"]),
        "learned_endpoint_mae_pct": float(recommended["endpoint_mae_pct"]),
        "learned_path_mae": float(recommended["path_mae"]),
        "learned_path_mae_pct": float(recommended["path_mae_pct"]),
        "learned_phase_mean_mae": float(recommended["phase_mean_mae"]),
        "learned_phase_mean_mae_pct": float(recommended["phase_mean_mae_pct"]),
        "learned_next_lock_mae": float(recommended.get("next_lock_mae", float("nan"))),
        "learned_next_lock_mae_pct": float(recommended.get("next_lock_mae_pct", float("nan"))),
        "learned_profile_shape_mae": float(recommended.get("profile_shape_mae", float("nan"))),
        "learned_profile_shape_mae_pct": float(
            recommended.get("profile_shape_mae_pct", float("nan"))
        ),
        "learned_energy_mae": float(recommended.get("energy_mae", float("nan"))),
        "learned_energy_mae_pct": float(recommended.get("energy_mae_pct", float("nan"))),
        "persistence_endpoint_mae": float(recommended["persistence_endpoint_mae"]),
        "persistence_endpoint_mae_pct": float(recommended["persistence_endpoint_mae_pct"]),
        "persistence_path_mae": float(recommended["persistence_path_mae"]),
        "persistence_path_mae_pct": float(recommended["persistence_path_mae_pct"]),
        "persistence_phase_mean_mae": float(recommended["persistence_phase_mean_mae"]),
        "persistence_phase_mean_mae_pct": float(recommended["persistence_phase_mean_mae_pct"]),
        "persistence_next_lock_mae": float(recommended.get("persistence_next_lock_mae", float("nan"))),
        "persistence_next_lock_mae_pct": float(
            recommended.get("persistence_next_lock_mae_pct", float("nan"))
        ),
        "persistence_profile_shape_mae": float(
            recommended.get("persistence_profile_shape_mae", float("nan"))
        ),
        "persistence_profile_shape_mae_pct": float(
            recommended.get("persistence_profile_shape_mae_pct", float("nan"))
        ),
        "best_baseline_endpoint_label": str(recommended["best_baseline_endpoint_label"]),
        "best_baseline_endpoint_mae": float(recommended["best_baseline_endpoint_mae"]),
        "best_baseline_endpoint_mae_pct": float(recommended["best_baseline_endpoint_mae_pct"]),
        "best_baseline_path_label": str(recommended["best_baseline_path_label"]),
        "best_baseline_path_mae": float(recommended["best_baseline_path_mae"]),
        "best_baseline_path_mae_pct": float(recommended["best_baseline_path_mae_pct"]),
        "best_baseline_phase_label": str(recommended["best_baseline_phase_label"]),
        "best_baseline_phase_mae": float(recommended["best_baseline_phase_mae"]),
        "best_baseline_phase_mae_pct": float(recommended["best_baseline_phase_mae_pct"]),
        "best_baseline_next_lock_label": str(recommended.get("best_baseline_next_lock_label", "")),
        "best_baseline_next_lock_mae": float(
            recommended.get("best_baseline_next_lock_mae", float("nan"))
        ),
        "best_baseline_next_lock_mae_pct": float(
            recommended.get("best_baseline_next_lock_mae_pct", float("nan"))
        ),
        "best_baseline_profile_shape_label": str(
            recommended.get("best_baseline_profile_shape_label", "")
        ),
        "best_baseline_profile_shape_mae": float(
            recommended.get("best_baseline_profile_shape_mae", float("nan"))
        ),
        "best_baseline_profile_shape_mae_pct": float(
            recommended.get("best_baseline_profile_shape_mae_pct", float("nan"))
        ),
        "endpoint_ratio_to_persistence": _safe_ratio(
            float(recommended["endpoint_mae"]),
            float(recommended["persistence_endpoint_mae"]),
        ),
        "path_ratio_to_persistence": _safe_ratio(
            float(recommended["path_mae"]),
            float(recommended["persistence_path_mae"]),
        ),
        "phase_ratio_to_persistence": _safe_ratio(
            float(recommended["phase_mean_mae"]),
            float(recommended["persistence_phase_mean_mae"]),
        ),
        "next_lock_ratio_to_persistence": _safe_ratio(
            float(recommended.get("next_lock_mae", float("nan"))),
            float(recommended.get("persistence_next_lock_mae", float("nan"))),
        ),
        "profile_shape_ratio_to_persistence": _safe_ratio(
            float(recommended.get("profile_shape_mae", float("nan"))),
            float(recommended.get("persistence_profile_shape_mae", float("nan"))),
        ),
        "endpoint_ratio_to_best_baseline": _safe_ratio(
            float(recommended["endpoint_mae"]),
            float(recommended["best_baseline_endpoint_mae"]),
        ),
        "path_ratio_to_best_baseline": _safe_ratio(
            float(recommended["path_mae"]),
            float(recommended["best_baseline_path_mae"]),
        ),
        "phase_ratio_to_best_baseline": _safe_ratio(
            float(recommended["phase_mean_mae"]),
            float(recommended["best_baseline_phase_mae"]),
        ),
        "next_lock_ratio_to_best_baseline": _safe_ratio(
            float(recommended.get("next_lock_mae", float("nan"))),
            float(recommended.get("best_baseline_next_lock_mae", float("nan"))),
        ),
        "profile_shape_ratio_to_best_baseline": _safe_ratio(
            float(recommended.get("profile_shape_mae", float("nan"))),
            float(recommended.get("best_baseline_profile_shape_mae", float("nan"))),
        ),
        "beats_persistence_endpoint": bool(recommended["beats_persistence_endpoint"]),
        "beats_persistence_path": bool(recommended["beats_persistence_path"]),
        "beats_persistence_phase": bool(recommended["beats_persistence_phase"]),
        "beats_persistence_next_lock": bool(recommended.get("beats_persistence_next_lock", False)),
        "beats_persistence_profile_shape": bool(
            recommended.get("beats_persistence_profile_shape", False)
        ),
        "beats_best_baseline_endpoint": bool(recommended["beats_best_baseline_endpoint"]),
        "beats_best_baseline_path": bool(recommended["beats_best_baseline_path"]),
        "beats_best_baseline_phase": bool(recommended["beats_best_baseline_phase"]),
        "beats_best_baseline_next_lock": bool(
            recommended.get("beats_best_baseline_next_lock", False)
        ),
        "beats_best_baseline_profile_shape": bool(
            recommended.get("beats_best_baseline_profile_shape", False)
        ),
        "support_count": int(recommended["origin_n"]),
        "support_type": "origins",
        "requested_origin_policy": str(origin_policy),
    }


def _curve_row_from_registry(
    *,
    registry_row: pd.Series,
    selection_target: str,
    origin_policy: str,
) -> dict[str, Any]:
    """Normalize one saved registry winner into the H5 curve schema."""
    return {
        "horizon_minutes": int(registry_row["requested_horizon_minutes"]),
        "source_stage": "007_rollout_challenger_sweep",
        "selection_policy": "challenger_sweep_registry",
        "selection_target": str(selection_target),
        "recommended_origin_policy": str(registry_row["recommended_origin_policy"]),
        "candidate_label": str(registry_row["recommended_candidate_label"]),
        "resolution": str(registry_row["recommended_resolution"]),
        "feature_set": str(registry_row["recommended_feature_set"]),
        "model_label": str(registry_row["recommended_model_label"]),
        "selection_source": str(registry_row["recommended_candidate_path"]),
        "learned_endpoint_mae": float(registry_row["endpoint_mae"]),
        "learned_endpoint_mae_pct": float(registry_row["endpoint_mae_pct"]),
        "learned_path_mae": float(registry_row["path_mae"]),
        "learned_path_mae_pct": float(registry_row["path_mae_pct"]),
        "learned_phase_mean_mae": float(registry_row["phase_mean_mae"]),
        "learned_phase_mean_mae_pct": float(registry_row["phase_mean_mae_pct"]),
        "learned_next_lock_mae": float(registry_row.get("next_lock_mae", float("nan"))),
        "learned_next_lock_mae_pct": float(registry_row.get("next_lock_mae_pct", float("nan"))),
        "learned_profile_shape_mae": float(registry_row.get("profile_shape_mae", float("nan"))),
        "learned_profile_shape_mae_pct": float(
            registry_row.get("profile_shape_mae_pct", float("nan"))
        ),
        "learned_energy_mae": float(registry_row.get("energy_mae", float("nan"))),
        "learned_energy_mae_pct": float(registry_row.get("energy_mae_pct", float("nan"))),
        "persistence_endpoint_mae": float(registry_row["persistence_endpoint_mae"]),
        "persistence_endpoint_mae_pct": float(registry_row["persistence_endpoint_mae_pct"]),
        "persistence_path_mae": float(registry_row["persistence_path_mae"]),
        "persistence_path_mae_pct": float(registry_row["persistence_path_mae_pct"]),
        "persistence_phase_mean_mae": float(registry_row["persistence_phase_mean_mae"]),
        "persistence_phase_mean_mae_pct": float(registry_row["persistence_phase_mean_mae_pct"]),
        "persistence_next_lock_mae": float(
            registry_row.get("persistence_next_lock_mae", float("nan"))
        ),
        "persistence_next_lock_mae_pct": float(
            registry_row.get("persistence_next_lock_mae_pct", float("nan"))
        ),
        "persistence_profile_shape_mae": float(
            registry_row.get("persistence_profile_shape_mae", float("nan"))
        ),
        "persistence_profile_shape_mae_pct": float(
            registry_row.get("persistence_profile_shape_mae_pct", float("nan"))
        ),
        "best_baseline_endpoint_label": str(registry_row["best_baseline_endpoint_label"]),
        "best_baseline_endpoint_mae": float(registry_row["best_baseline_endpoint_mae"]),
        "best_baseline_endpoint_mae_pct": float(registry_row["best_baseline_endpoint_mae_pct"]),
        "best_baseline_path_label": str(registry_row["best_baseline_path_label"]),
        "best_baseline_path_mae": float(registry_row["best_baseline_path_mae"]),
        "best_baseline_path_mae_pct": float(registry_row["best_baseline_path_mae_pct"]),
        "best_baseline_phase_label": str(registry_row["best_baseline_phase_label"]),
        "best_baseline_phase_mae": float(registry_row["best_baseline_phase_mae"]),
        "best_baseline_phase_mae_pct": float(registry_row["best_baseline_phase_mae_pct"]),
        "best_baseline_next_lock_label": str(registry_row.get("best_baseline_next_lock_label", "")),
        "best_baseline_next_lock_mae": float(
            registry_row.get("best_baseline_next_lock_mae", float("nan"))
        ),
        "best_baseline_next_lock_mae_pct": float(
            registry_row.get("best_baseline_next_lock_mae_pct", float("nan"))
        ),
        "best_baseline_profile_shape_label": str(
            registry_row.get("best_baseline_profile_shape_label", "")
        ),
        "best_baseline_profile_shape_mae": float(
            registry_row.get("best_baseline_profile_shape_mae", float("nan"))
        ),
        "best_baseline_profile_shape_mae_pct": float(
            registry_row.get("best_baseline_profile_shape_mae_pct", float("nan"))
        ),
        "endpoint_ratio_to_persistence": _safe_ratio(
            float(registry_row["endpoint_mae"]),
            float(registry_row["persistence_endpoint_mae"]),
        ),
        "path_ratio_to_persistence": _safe_ratio(
            float(registry_row["path_mae"]),
            float(registry_row["persistence_path_mae"]),
        ),
        "phase_ratio_to_persistence": _safe_ratio(
            float(registry_row["phase_mean_mae"]),
            float(registry_row["persistence_phase_mean_mae"]),
        ),
        "next_lock_ratio_to_persistence": _safe_ratio(
            float(registry_row.get("next_lock_mae", float("nan"))),
            float(registry_row.get("persistence_next_lock_mae", float("nan"))),
        ),
        "profile_shape_ratio_to_persistence": _safe_ratio(
            float(registry_row.get("profile_shape_mae", float("nan"))),
            float(registry_row.get("persistence_profile_shape_mae", float("nan"))),
        ),
        "endpoint_ratio_to_best_baseline": _safe_ratio(
            float(registry_row["endpoint_mae"]),
            float(registry_row["best_baseline_endpoint_mae"]),
        ),
        "path_ratio_to_best_baseline": _safe_ratio(
            float(registry_row["path_mae"]),
            float(registry_row["best_baseline_path_mae"]),
        ),
        "phase_ratio_to_best_baseline": _safe_ratio(
            float(registry_row["phase_mean_mae"]),
            float(registry_row["best_baseline_phase_mae"]),
        ),
        "next_lock_ratio_to_best_baseline": _safe_ratio(
            float(registry_row.get("next_lock_mae", float("nan"))),
            float(registry_row.get("best_baseline_next_lock_mae", float("nan"))),
        ),
        "profile_shape_ratio_to_best_baseline": _safe_ratio(
            float(registry_row.get("profile_shape_mae", float("nan"))),
            float(registry_row.get("best_baseline_profile_shape_mae", float("nan"))),
        ),
        "beats_persistence_endpoint": bool(registry_row["beats_persistence_endpoint"]),
        "beats_persistence_path": bool(registry_row["beats_persistence_path"]),
        "beats_persistence_phase": bool(registry_row["beats_persistence_phase"]),
        "beats_persistence_next_lock": bool(registry_row.get("beats_persistence_next_lock", False)),
        "beats_persistence_profile_shape": bool(
            registry_row.get("beats_persistence_profile_shape", False)
        ),
        "beats_best_baseline_endpoint": bool(registry_row["beats_best_baseline_endpoint"]),
        "beats_best_baseline_path": bool(registry_row["beats_best_baseline_path"]),
        "beats_best_baseline_phase": bool(registry_row["beats_best_baseline_phase"]),
        "beats_best_baseline_next_lock": bool(
            registry_row.get("beats_best_baseline_next_lock", False)
        ),
        "beats_best_baseline_profile_shape": bool(
            registry_row.get("beats_best_baseline_profile_shape", False)
        ),
        "support_count": int(registry_row["origin_n"]),
        "support_type": "origins",
        "requested_origin_policy": str(origin_policy),
    }


def _load_candidate_frame_from_registry_row(registry_row: pd.Series) -> pd.DataFrame:
    """Load the candidate-results artifact referenced by a registry row."""
    candidate_results_path = _project_artifact_path(str(registry_row["candidate_results_path"]))
    if not candidate_results_path.exists():
        raise FileNotFoundError(
            f"Missing challenger candidate-results artifact referenced by Stage-8 registry: {candidate_results_path}"
        )
    frame = pd.read_csv(candidate_results_path)
    frame.insert(0, "requested_horizon_minutes", int(registry_row["requested_horizon_minutes"]))
    return frame


def _origin_policy_matches(registry_row: pd.Series, requested_origin_policy: str) -> bool:
    """Return whether a saved registry row is comparable to the requested policy."""
    recommended_origin_policy = registry_row.get("recommended_origin_policy", pd.NA)
    if pd.isna(recommended_origin_policy):
        return False
    return str(recommended_origin_policy) == str(requested_origin_policy)


def _plot_ratio_curve(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot learned-to-persistence MAE ratios across forecast horizons."""
    plot_df = summary.sort_values("horizon_minutes").copy()
    horizons = plot_df["horizon_minutes"].to_numpy(dtype=float)
    plt.figure(figsize=(10, 6))
    for column, marker, label in (
        ("endpoint_ratio_to_persistence", "o", "Endpoint MAE / Persistence MAE"),
        ("path_ratio_to_persistence", "s", "Path MAE / Persistence MAE"),
        ("phase_ratio_to_persistence", "^", "Phase MAE / Persistence MAE"),
        ("next_lock_ratio_to_persistence", "D", "Next-lock MAE / Persistence MAE"),
        ("profile_shape_ratio_to_persistence", "v", "Profile-shape MAE / Persistence MAE"),
    ):
        if column not in plot_df.columns:
            continue
        values = plot_df[column].to_numpy(dtype=float)
        if not np.isfinite(values).any():
            continue
        plt.plot(
            horizons,
            values,
            marker=marker,
            linewidth=2,
            label=label,
        )
    plt.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Parity with persistence")
    plt.xscale("log")
    plt.xticks(horizons, [f"{int(value)}m" for value in horizons], rotation=45)
    plt.xlabel("Forecast horizon (minutes)")
    plt.ylabel("Learned / Persistence MAE ratio")
    plt.title("H5 horizon degradation curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    validate_png_artifact(output_path)


def _plot_absolute_curve(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot absolute learned, baseline, and persistence MAE surfaces by horizon."""
    plot_df = summary.sort_values("horizon_minutes").copy()
    horizons = plot_df["horizon_minutes"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axis_specs = [
        (
            axes[0, 0],
            "learned_endpoint_mae",
            "best_baseline_endpoint_mae",
            "persistence_endpoint_mae",
            "Endpoint MAE by horizon",
        ),
        (
            axes[0, 1],
            "learned_path_mae",
            "best_baseline_path_mae",
            "persistence_path_mae",
            "Path MAE by horizon",
        ),
        (
            axes[0, 2],
            "learned_phase_mean_mae",
            "best_baseline_phase_mae",
            "persistence_phase_mean_mae",
            "Phase-average MAE by horizon",
        ),
        (
            axes[1, 0],
            "learned_next_lock_mae",
            "best_baseline_next_lock_mae",
            "persistence_next_lock_mae",
            "Next-lock MAE by horizon",
        ),
        (
            axes[1, 1],
            "learned_profile_shape_mae",
            "best_baseline_profile_shape_mae",
            "persistence_profile_shape_mae",
            "Profile-shape MAE by horizon",
        ),
        (
            axes[1, 2],
            "learned_energy_mae",
            None,
            None,
            "Energy-bias MAE by horizon",
        ),
    ]
    for axis, learned_col, baseline_col, persistence_col, title in axis_specs:
        if learned_col not in plot_df.columns:
            axis.axis("off")
            continue
        learned_values = plot_df[learned_col].to_numpy(dtype=float)
        if not np.isfinite(learned_values).any():
            axis.axis("off")
            continue
        axis.plot(horizons, learned_values, marker="o", linewidth=2, label="Learned")
        if baseline_col is not None and baseline_col in plot_df.columns:
            baseline_values = plot_df[baseline_col].to_numpy(dtype=float)
            if np.isfinite(baseline_values).any():
                axis.plot(horizons, baseline_values, marker="^", linewidth=2, label="Best baseline")
        if persistence_col is not None and persistence_col in plot_df.columns:
            persistence_values = plot_df[persistence_col].to_numpy(dtype=float)
            if np.isfinite(persistence_values).any():
                axis.plot(horizons, persistence_values, marker="s", linewidth=2, label="Persistence")
        axis.set_title(title)
        axis.set_ylabel("MAE")
        axis.grid(True, alpha=0.3)
        axis.legend()
        axis.set_xscale("log")
        axis.set_xticks(horizons)
        axis.set_xticklabels([f"{int(value)}m" for value in horizons], rotation=45)
        axis.set_xlabel("Forecast horizon (minutes)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    validate_png_artifact(output_path)


def _write_curve_summary_md(summary: pd.DataFrame, output_path: Path) -> None:
    """Write the human-readable markdown companion for the H5 summary table."""
    beats_path = summary.loc[summary["beats_persistence_path"], "horizon_minutes"].astype(int).tolist()
    beats_endpoint = summary.loc[summary["beats_persistence_endpoint"], "horizon_minutes"].astype(int).tolist()
    beats_phase = summary.loc[summary["beats_persistence_phase"], "horizon_minutes"].astype(int).tolist()
    beats_next_lock = summary.loc[
        summary["beats_persistence_next_lock"], "horizon_minutes"
    ].astype(int).tolist()
    beats_profile_shape = summary.loc[
        summary["beats_persistence_profile_shape"], "horizon_minutes"
    ].astype(int).tolist()
    beats_baseline_path = summary.loc[summary["beats_best_baseline_path"], "horizon_minutes"].astype(int).tolist()
    beats_baseline_endpoint = summary.loc[
        summary["beats_best_baseline_endpoint"], "horizon_minutes"
    ].astype(int).tolist()
    beats_baseline_phase = summary.loc[
        summary["beats_best_baseline_phase"], "horizon_minutes"
    ].astype(int).tolist()
    beats_baseline_next_lock = summary.loc[
        summary["beats_best_baseline_next_lock"], "horizon_minutes"
    ].astype(int).tolist()
    beats_baseline_profile_shape = summary.loc[
        summary["beats_best_baseline_profile_shape"], "horizon_minutes"
    ].astype(int).tolist()
    best_path = summary.sort_values(
        ["path_ratio_to_persistence", "horizon_minutes"], ascending=[True, True], kind="stable"
    ).iloc[0]
    best_endpoint = summary.sort_values(
        ["endpoint_ratio_to_persistence", "horizon_minutes"], ascending=[True, True], kind="stable"
    ).iloc[0]
    best_phase = summary.sort_values(
        ["phase_ratio_to_persistence", "horizon_minutes"], ascending=[True, True], kind="stable"
    ).iloc[0]
    best_next_lock = summary.sort_values(
        ["next_lock_ratio_to_persistence", "horizon_minutes"], ascending=[True, True], kind="stable"
    ).iloc[0]
    best_profile_shape = summary.sort_values(
        ["profile_shape_ratio_to_persistence", "horizon_minutes"],
        ascending=[True, True],
        kind="stable",
    ).iloc[0]

    lines = [
        "# H5 Horizon Degradation Curve",
        "",
        "- Methodology: `1m` uses the current Stage-5 holdout learned candidate; longer horizons use Stage-7 challenger sweeps.",
        "- Anchor note: the `1m` Stage-5 row uses the single-step holdout MAE as the fallback value for next-lock and profile-shape metrics so the curve stays continuous.",
        "- Interpretation note: this is a capability envelope, not a single-model monotonic decay trace. Candidate selection can change by horizon.",
        f"- Best path ratio to persistence: `{best_path['path_ratio_to_persistence']:.4f}` at `{int(best_path['horizon_minutes'])}m` using `{best_path['candidate_label']}`.",
        f"- Best endpoint ratio to persistence: `{best_endpoint['endpoint_ratio_to_persistence']:.4f}` at `{int(best_endpoint['horizon_minutes'])}m` using `{best_endpoint['candidate_label']}`.",
        f"- Best phase ratio to persistence: `{best_phase['phase_ratio_to_persistence']:.4f}` at `{int(best_phase['horizon_minutes'])}m` using `{best_phase['candidate_label']}`.",
        f"- Best next-lock ratio to persistence: `{best_next_lock['next_lock_ratio_to_persistence']:.4f}` at `{int(best_next_lock['horizon_minutes'])}m` using `{best_next_lock['candidate_label']}`.",
        f"- Best profile-shape ratio to persistence: `{best_profile_shape['profile_shape_ratio_to_persistence']:.4f}` at `{int(best_profile_shape['horizon_minutes'])}m` using `{best_profile_shape['candidate_label']}`.",
        f"- Horizons beating persistence on path MAE: `{beats_path}`.",
        f"- Horizons beating persistence on endpoint MAE: `{beats_endpoint}`.",
        f"- Horizons beating persistence on phase-average MAE: `{beats_phase}`.",
        f"- Horizons beating persistence on next-lock MAE: `{beats_next_lock}`.",
        f"- Horizons beating persistence on profile-shape MAE: `{beats_profile_shape}`.",
        f"- Horizons beating the best baseline on path MAE: `{beats_baseline_path}`.",
        f"- Horizons beating the best baseline on endpoint MAE: `{beats_baseline_endpoint}`.",
        f"- Horizons beating the best baseline on phase-average MAE: `{beats_baseline_phase}`.",
        f"- Horizons beating the best baseline on next-lock MAE: `{beats_baseline_next_lock}`.",
        f"- Horizons beating the best baseline on profile-shape MAE: `{beats_baseline_profile_shape}`.",
        "",
        "## Horizon Summary",
        "",
    ]
    for _, row in summary.iterrows():
        recommended_origin_policy = row.get("recommended_origin_policy", row["requested_origin_policy"])
        if pd.isna(recommended_origin_policy) or not str(recommended_origin_policy):
            recommended_origin_policy = row["requested_origin_policy"]
        lines.extend(
            [
                f"### {int(row['horizon_minutes'])}m",
                f"- Candidate: `{row['candidate_label']}`",
                f"- Source: `{row['source_stage']}` via `{row['selection_policy']}`",
                f"- Recommended origin policy: `{recommended_origin_policy}`",
                f"- Learned endpoint/path/phase MAE: `{float(row['learned_endpoint_mae']):.6f}` ({float(row['learned_endpoint_mae_pct']):.3f}%) / `{float(row['learned_path_mae']):.6f}` ({float(row['learned_path_mae_pct']):.3f}%) / `{float(row['learned_phase_mean_mae']):.6f}` ({float(row['learned_phase_mean_mae_pct']):.3f}%)",
                f"- Learned next-lock/profile-shape MAE: `{float(row['learned_next_lock_mae']):.6f}` ({float(row['learned_next_lock_mae_pct']):.3f}%) / `{float(row['learned_profile_shape_mae']):.6f}` ({float(row['learned_profile_shape_mae_pct']):.3f}%)",
                f"- Persistence endpoint/path/phase MAE: `{float(row['persistence_endpoint_mae']):.6f}` ({float(row['persistence_endpoint_mae_pct']):.3f}%) / `{float(row['persistence_path_mae']):.6f}` ({float(row['persistence_path_mae_pct']):.3f}%) / `{float(row['persistence_phase_mean_mae']):.6f}` ({float(row['persistence_phase_mean_mae_pct']):.3f}%)",
                f"- Persistence next-lock/profile-shape MAE: `{float(row['persistence_next_lock_mae']):.6f}` ({float(row['persistence_next_lock_mae_pct']):.3f}%) / `{float(row['persistence_profile_shape_mae']):.6f}` ({float(row['persistence_profile_shape_mae_pct']):.3f}%)",
                f"- Best baseline endpoint/path/phase: `{row['best_baseline_endpoint_label']}` / `{row['best_baseline_path_label']}` / `{row['best_baseline_phase_label']}`",
                f"- Best baseline next-lock/profile-shape: `{row['best_baseline_next_lock_label']}` / `{row['best_baseline_profile_shape_label']}`",
                f"- Ratios to persistence (endpoint/path/phase): `{float(row['endpoint_ratio_to_persistence']):.4f}` / `{float(row['path_ratio_to_persistence']):.4f}` / `{float(row['phase_ratio_to_persistence']):.4f}`",
                f"- Ratios to persistence (next-lock/profile-shape): `{float(row['next_lock_ratio_to_persistence']):.4f}` / `{float(row['profile_shape_ratio_to_persistence']):.4f}`",
                "",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_horizon_curve(
    *,
    output_root: Path,
    horizons_minutes: list[int],
    origins: int,
    origin_policy: str,
    selection_target: str,
    max_candidates: int,
    include_stage5_anchor: bool,
    reuse_existing_sweeps: bool,
) -> dict[str, Any]:
    """Build the Stage-8 horizon curve and persist its summary artifacts."""
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    unique_horizons = sorted({int(value) for value in horizons_minutes})
    summary_rows: list[dict[str, Any]] = []
    candidate_frames: list[pd.DataFrame] = []

    if include_stage5_anchor and 1 in unique_horizons:
        summary_rows.append(_load_stage5_nowcast_anchor())
        unique_horizons = [value for value in unique_horizons if value != 1]

    rollout_output_root = scoped_output_path(PATHS["outputs_rollout_dir"])
    sweep_registry = (
        _build_challenger_sweep_registry_snapshot(rollout_output_root)
        if reuse_existing_sweeps
        else pd.DataFrame()
    )
    for horizon_minutes in unique_horizons:
        resolved_origin_policy = resolve_rollout_origin_policy(
            int(horizon_minutes),
            str(origin_policy),
        )
        resolved_selection_target = resolve_rollout_selection_target(
            int(horizon_minutes),
            str(selection_target),
        )
        registry_row = (
            _select_challenger_sweep_registry_candidate(
                sweep_registry,
                requested_horizon_minutes=int(horizon_minutes),
                requested_origin_policy=resolved_origin_policy,
                selection_target=resolved_selection_target,
            )
            if reuse_existing_sweeps
            else None
        )
        if registry_row is not None and not _origin_policy_matches(registry_row, resolved_origin_policy):
            logger.info(
                "Ignoring registry sweep %s for %sm because it was measured under origin_policy=%s instead of %s",
                registry_row["sweep_run_id"],
                horizon_minutes,
                registry_row.get("recommended_origin_policy"),
                resolved_origin_policy,
            )
            registry_row = None
        if registry_row is not None:
            logger.info(
                "Reusing measured Stage-7 challenger sweep %s for %sm horizon characterization",
                registry_row["sweep_run_id"],
                horizon_minutes,
            )
            summary_rows.append(
                _curve_row_from_registry(
                    registry_row=registry_row,
                    selection_target=resolved_selection_target,
                    origin_policy=resolved_origin_policy,
                )
            )
            candidate_frames.append(_load_candidate_frame_from_registry_row(registry_row))
            continue

        logger.info("Running H5 horizon characterization for %sm", horizon_minutes)
        sweep_result = run_rollout_challenger_sweep(
            output_root=rollout_output_root,
            horizon_minutes=int(horizon_minutes),
            origins=int(origins),
            origin_policy=resolved_origin_policy,
            selection_target=resolved_selection_target,
            max_candidates=int(max_candidates),
            refresh_rollout_registry=True,
            refresh_rollout_latest=int(horizon_minutes) == int(MULTIRES_ROLLOUT["horizon_minutes"]),
            refresh_sweep_latest=False,
        )
        summary_rows.append(
            _curve_row_from_sweep(
                horizon_minutes=int(horizon_minutes),
                selection_target=resolved_selection_target,
                origin_policy=resolved_origin_policy,
                sweep_result=sweep_result,
            )
        )
        candidate_results = sweep_result["candidate_results"].copy()
        candidate_results.insert(0, "requested_horizon_minutes", int(horizon_minutes))
        candidate_frames.append(candidate_results)
        if reuse_existing_sweeps:
            sweep_registry = _build_challenger_sweep_registry_snapshot(rollout_output_root)

    summary = pd.DataFrame(summary_rows).sort_values("horizon_minutes").reset_index(drop=True)
    candidate_results_all = (
        pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    )
    summary.to_csv(run_dir / "horizon_curve_summary.csv", index=False, float_format="%.6f")
    candidate_results_all.to_csv(
        run_dir / "horizon_curve_candidates.csv", index=False, float_format="%.6f"
    )

    _plot_ratio_curve(summary, run_dir / "fig_horizon_ratio_curve.png")
    _plot_absolute_curve(summary, run_dir / "fig_horizon_absolute_mae.png")
    _write_curve_summary_md(summary, run_dir / "horizon_curve_summary.md")
    write_figure_guide(
        output_path=run_dir / "figure_guide.md",
        stage_title="Stage-8 Horizon-Curve Figures",
        stage_purpose=(
            "These figures explain where the current stack wins or loses as forecast "
            "horizon grows. The horizon curve is a capability envelope, not a single-model decay chart."
        ),
        figures=[
            FigureGuideEntry(
                filename="fig_horizon_ratio_curve.png",
                title="Ratio curve",
                intent="Show learned performance relative to persistence across each horizon and objective.",
                how_to_read="A value below 1.0 means the learned candidate beat persistence on that metric at that horizon.",
                look_for="Crossings above or below parity, especially where next-lock quality improves before profile-shape quality does.",
            ),
            FigureGuideEntry(
                filename="fig_horizon_absolute_mae.png",
                title="Absolute MAE surface",
                intent="Show the absolute error tradeoffs among the learned winner, the best baseline, and persistence.",
                how_to_read="Each subplot focuses on one metric family across horizons; compare the learned line to the baseline and persistence lines.",
                look_for="Horizons where the learned candidate beats persistence but still trails a stronger baseline, and horizons where it becomes the clear winner.",
            ),
        ],
    )

    crossover_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "load_type": DATASET["load_type"],
        "artifact_namespace": DATASET["artifact_namespace"],
        "selection_target": str(selection_target),
        "horizons_minutes": [int(value) for value in summary["horizon_minutes"].tolist()],
        "beats_persistence_path_horizons": (
            summary.loc[summary["beats_persistence_path"], "horizon_minutes"].astype(int).tolist()
        ),
        "beats_persistence_endpoint_horizons": (
            summary.loc[summary["beats_persistence_endpoint"], "horizon_minutes"].astype(int).tolist()
        ),
        "beats_persistence_phase_horizons": (
            summary.loc[summary["beats_persistence_phase"], "horizon_minutes"].astype(int).tolist()
        ),
        "beats_persistence_next_lock_horizons": (
            summary.loc[summary["beats_persistence_next_lock"], "horizon_minutes"]
            .astype(int)
            .tolist()
        ),
        "beats_persistence_profile_shape_horizons": (
            summary.loc[summary["beats_persistence_profile_shape"], "horizon_minutes"]
            .astype(int)
            .tolist()
        ),
        "beats_best_baseline_path_horizons": (
            summary.loc[summary["beats_best_baseline_path"], "horizon_minutes"].astype(int).tolist()
        ),
        "beats_best_baseline_endpoint_horizons": (
            summary.loc[summary["beats_best_baseline_endpoint"], "horizon_minutes"].astype(int).tolist()
        ),
        "beats_best_baseline_phase_horizons": (
            summary.loc[summary["beats_best_baseline_phase"], "horizon_minutes"].astype(int).tolist()
        ),
        "beats_best_baseline_next_lock_horizons": (
            summary.loc[summary["beats_best_baseline_next_lock"], "horizon_minutes"]
            .astype(int)
            .tolist()
        ),
        "beats_best_baseline_profile_shape_horizons": (
            summary.loc[summary["beats_best_baseline_profile_shape"], "horizon_minutes"]
            .astype(int)
            .tolist()
        ),
        "note": (
            "Ratios can be non-monotonic because the best verified learned candidate is allowed "
            "to change by horizon. Treat this as a capability envelope, not a single-model decay trace."
        ),
    }
    (run_dir / "crossover_summary.json").write_text(
        json.dumps(crossover_payload, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "run_id": run_dir.name,
        "stage": "008_horizon_curve",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "load_type": DATASET["load_type"],
        "artifact_namespace": DATASET["artifact_namespace"],
        "horizons_minutes": [int(value) for value in summary["horizon_minutes"].tolist()],
        "selection_target": str(selection_target),
        "origin_policy": str(origin_policy),
        "origins_per_run": int(origins),
        "max_candidates": int(max_candidates),
        "include_stage5_anchor": bool(include_stage5_anchor),
        "reuse_existing_sweeps": bool(reuse_existing_sweeps),
        "config_hash": stable_config_hash(
            {
                "horizons_minutes": [int(value) for value in summary["horizon_minutes"].tolist()],
                "selection_target": str(selection_target),
                "origin_policy": str(origin_policy),
                "origins_per_run": int(origins),
                "max_candidates": int(max_candidates),
                "include_stage5_anchor": bool(include_stage5_anchor),
                "reuse_existing_sweeps": bool(reuse_existing_sweeps),
            }
        ),
        "artifacts": {
            "horizon_curve_summary": "horizon_curve_summary.csv",
            "horizon_curve_candidates": "horizon_curve_candidates.csv",
            "horizon_curve_summary_md": "horizon_curve_summary.md",
            "crossover_summary": "crossover_summary.json",
            "figure_guide_md": "figure_guide.md",
            "fig_horizon_ratio_curve": "fig_horizon_ratio_curve.png",
            "fig_horizon_absolute_mae": "fig_horizon_absolute_mae.png",
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    update_latest_alias(run_dir, output_root / "latest", enabled=bool(MULTIRES_HORIZON_CURVE["write_latest"]))
    logger.info("H5 horizon-curve artifacts written to %s", run_dir)
    return {"run_dir": run_dir, "summary": summary, "candidates": candidate_results_all, "manifest": manifest}


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the Stage-8 horizon-curve runner."""
    parser = argparse.ArgumentParser(description="Build the H5 horizon degradation curve.")
    parser.add_argument("--horizon-minutes", type=int, action="append", dest="horizons_minutes")
    parser.add_argument(
        "--origins",
        type=int,
        default=MULTIRES_HORIZON_CURVE["origins_per_run"],
    )
    parser.add_argument(
        "--origin-policy",
        choices=["uniform", "midnight", "billing_aligned", "phase_balanced", "auto"],
        default=MULTIRES_HORIZON_CURVE["origin_policy"],
    )
    parser.add_argument(
        "--selection-target",
        choices=[
            "endpoint_mae",
            "path_mae",
            "phase_mean_mae",
            "next_lock_mae",
            "profile_shape_mae",
            "auto",
        ],
        default=MULTIRES_HORIZON_CURVE["selection_target"],
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=MULTIRES_HORIZON_CURVE["max_candidates"],
    )
    parser.add_argument(
        "--skip-stage5-anchor",
        action="store_true",
        help="Skip the 1-minute Stage-5 holdout anchor row even if 1m is requested.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore the challenger sweep registry and rerun fresh Stage-7 sweeps for every requested horizon.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(scoped_output_path(PATHS["outputs_horizon_curve_dir"])),
    )
    return parser.parse_args()


def main() -> int:
    """Execute Stage-8 and write one timestamped horizon-curve run directory."""
    _configure_logging()
    validate_config()
    args = parse_args()
    if not MULTIRES_HORIZON_CURVE["enabled"]:
        logger.error("multires.horizon_curve.enabled=false; horizon-curve execution is disabled.")
        return 1
    horizons_minutes = args.horizons_minutes or list(MULTIRES_HORIZON_CURVE["horizons_minutes"])
    try:
        run_horizon_curve(
            output_root=Path(args.output_dir).resolve(),
            horizons_minutes=[int(value) for value in horizons_minutes],
            origins=int(args.origins),
            origin_policy=str(args.origin_policy),
            selection_target=str(args.selection_target),
            max_candidates=int(args.max_candidates),
            include_stage5_anchor=not bool(args.skip_stage5_anchor),
            reuse_existing_sweeps=(
                bool(MULTIRES_HORIZON_CURVE["reuse_existing_sweeps"]) and not bool(args.force_refresh)
            ),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
