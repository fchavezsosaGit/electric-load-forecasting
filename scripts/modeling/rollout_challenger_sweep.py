"""Scheduled Stage-7 challenger sweeps driven by Stage-6 and Stage-7 registries.

This module evaluates multiple rollout candidates on a shared-origin surface so
the repository can promote the strongest measured challenger for a requested
horizon and objective without relying on mutable `latest/` aliases or ad hoc runs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

import sys

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import (  # noqa: E402
    DATASET,
    MULTIRES_CONFIG,
    MULTIRES_ROLLOUT,
    MULTIRES_ROLLOUT_CHALLENGERS,
    MULTIRES_ROLLOUT_SWEEP_POLICIES,
    PATHS,
    output_path_candidates,
    resolve_horizon_policy,
    resolve_rollout_origin_policy,
    resolve_rollout_selection_target,
    scoped_output_path,
    validate_config,
)
from modeling.common import (  # noqa: E402
    build_model_catalog,
    lead_steps_for_horizon,
    stable_config_hash,
    update_latest_alias,
)
from modeling.multires import load_base_gold  # noqa: E402
from modeling.multires_compare import _build_winner_registry_snapshot  # noqa: E402
from modeling.parallel import resolve_parallel_plan  # noqa: E402
from modeling.recursive_rollout import (  # noqa: E402
    _aggregate_rollout_metrics,
    _build_rollout_registry_snapshot,
    _coerce_bool_series,
    _enabled_rollout_baselines,
    _phase_bucket_seconds,
    _read_csv_if_present,
    _sample_rollout_origin_timestamps,
    _select_rollout_origins,
    _selection_metric_fields,
    run_rollout_evaluation,
)
from modeling.runtime import supports_high_capacity_parallelism  # noqa: E402

logger = logging.getLogger(__name__)
PROJECT_ROOT = SCRIPTS_DIR.parent
CHALLENGER_SWEEP_REGISTRY_COLUMNS = [
    "sweep_run_id",
    "generated_at_utc",
    "load_type",
    "artifact_namespace",
    "requested_horizon_minutes",
    "selection_target",
    "origin_selection_scope",
    "shared_origin_count",
    "recommended_origin_policy",
    "recommended_candidate_label",
    "recommended_resolution",
    "recommended_feature_set",
    "recommended_model_label",
    "recommended_target_mode",
    "recommended_source_type",
    "recommended_run_id",
    "recommended_run_path",
    "recommended_metric_value",
    "recommended_metric_pct",
    "endpoint_mae",
    "endpoint_mae_pct",
    "path_mae",
    "path_mae_pct",
    "phase_mean_mae",
    "phase_mean_mae_pct",
    "next_lock_mae",
    "next_lock_mae_pct",
    "profile_shape_mae",
    "profile_shape_mae_pct",
    "energy_mae",
    "energy_mae_pct",
    "mean_coverage",
    "origin_n",
    "persistence_endpoint_mae",
    "persistence_endpoint_mae_pct",
    "persistence_path_mae",
    "persistence_path_mae_pct",
    "persistence_phase_mean_mae",
    "persistence_phase_mean_mae_pct",
    "persistence_next_lock_mae",
    "persistence_next_lock_mae_pct",
    "persistence_profile_shape_mae",
    "persistence_profile_shape_mae_pct",
    "best_baseline_endpoint_label",
    "best_baseline_endpoint_mae",
    "best_baseline_endpoint_mae_pct",
    "best_baseline_path_label",
    "best_baseline_path_mae",
    "best_baseline_path_mae_pct",
    "best_baseline_phase_label",
    "best_baseline_phase_mae",
    "best_baseline_phase_mae_pct",
    "best_baseline_next_lock_label",
    "best_baseline_next_lock_mae",
    "best_baseline_next_lock_mae_pct",
    "best_baseline_profile_shape_label",
    "best_baseline_profile_shape_mae",
    "best_baseline_profile_shape_mae_pct",
    "beats_persistence_endpoint",
    "beats_persistence_path",
    "beats_persistence_phase",
    "beats_persistence_next_lock",
    "beats_persistence_profile_shape",
    "beats_best_baseline_endpoint",
    "beats_best_baseline_path",
    "beats_best_baseline_phase",
    "beats_best_baseline_next_lock",
    "beats_best_baseline_profile_shape",
    "candidate_results_path",
    "recommended_candidate_path",
    "sweep_path",
]


def _write_sweep_manifest(sweep_dir: Path, manifest: dict[str, Any]) -> None:
    """Persist the sweep manifest so partial runs remain inspectable on failure."""
    (sweep_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _configure_logging() -> None:
    """Initialize a simple logger for standalone challenger-sweep execution."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _relative_artifact_path(path: Path) -> str:
    """Render an artifact path relative to the repository root when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _build_challenger_sweep_registry_snapshot(output_root: Path) -> pd.DataFrame:
    """Scan completed challenger sweeps into a reusable horizon-selection registry."""
    sweep_root = Path(output_root).resolve() / "challenger_sweeps"
    rows: list[dict[str, Any]] = []
    if not sweep_root.exists():
        return pd.DataFrame(columns=CHALLENGER_SWEEP_REGISTRY_COLUMNS)

    for sweep_dir in sorted(sweep_root.iterdir()):
        if not sweep_dir.is_dir() or sweep_dir.name == "latest":
            continue
        recommended_path = sweep_dir / "recommended_candidate.json"
        candidate_results_path = sweep_dir / "candidate_results.csv"
        if not recommended_path.exists() or not candidate_results_path.exists():
            continue

        try:
            payload = json.loads(recommended_path.read_text(encoding="utf-8"))
            results = _read_csv_if_present(candidate_results_path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping malformed challenger sweep artifact %s: %s", sweep_dir, exc)
            continue
        manifest_path = sweep_dir / "run_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            manifest = {}
        if results.empty:
            continue
        if "phase_mean_mae" not in results.columns:
            results["phase_mean_mae"] = results.get("path_mae", float("nan"))
        if "phase_mean_mae_pct" not in results.columns:
            results["phase_mean_mae_pct"] = results.get("path_mae_pct", float("nan"))
        if "next_lock_mae" not in results.columns:
            results["next_lock_mae"] = results.get("path_mae", float("nan"))
        if "next_lock_mae_pct" not in results.columns:
            results["next_lock_mae_pct"] = results.get("path_mae_pct", float("nan"))
        if "profile_shape_mae" not in results.columns:
            results["profile_shape_mae"] = float("nan")
        if "profile_shape_mae_pct" not in results.columns:
            results["profile_shape_mae_pct"] = float("nan")
        if "energy_mae" not in results.columns:
            results["energy_mae"] = float("nan")
        if "energy_mae_pct" not in results.columns:
            results["energy_mae_pct"] = float("nan")
        if "persistence_phase_mean_mae" not in results.columns:
            results["persistence_phase_mean_mae"] = results.get("persistence_path_mae", float("nan"))
        if "persistence_phase_mean_mae_pct" not in results.columns:
            results["persistence_phase_mean_mae_pct"] = results.get(
                "persistence_path_mae_pct",
                float("nan"),
            )
        if "persistence_next_lock_mae" not in results.columns:
            results["persistence_next_lock_mae"] = results.get("persistence_path_mae", float("nan"))
        if "persistence_next_lock_mae_pct" not in results.columns:
            results["persistence_next_lock_mae_pct"] = results.get(
                "persistence_path_mae_pct",
                float("nan"),
            )
        if "persistence_profile_shape_mae" not in results.columns:
            results["persistence_profile_shape_mae"] = float("nan")
        if "persistence_profile_shape_mae_pct" not in results.columns:
            results["persistence_profile_shape_mae_pct"] = float("nan")
        if "best_baseline_phase_label" not in results.columns:
            results["best_baseline_phase_label"] = results.get("best_baseline_path_label", "")
        if "best_baseline_phase_mae" not in results.columns:
            results["best_baseline_phase_mae"] = results.get("best_baseline_path_mae", float("nan"))
        if "best_baseline_phase_mae_pct" not in results.columns:
            results["best_baseline_phase_mae_pct"] = results.get(
                "best_baseline_path_mae_pct",
                float("nan"),
            )
        if "best_baseline_next_lock_label" not in results.columns:
            results["best_baseline_next_lock_label"] = results.get("best_baseline_path_label", "")
        if "best_baseline_next_lock_mae" not in results.columns:
            results["best_baseline_next_lock_mae"] = results.get("best_baseline_path_mae", float("nan"))
        if "best_baseline_next_lock_mae_pct" not in results.columns:
            results["best_baseline_next_lock_mae_pct"] = results.get(
                "best_baseline_path_mae_pct",
                float("nan"),
            )
        if "best_baseline_profile_shape_label" not in results.columns:
            results["best_baseline_profile_shape_label"] = ""
        if "best_baseline_profile_shape_mae" not in results.columns:
            results["best_baseline_profile_shape_mae"] = float("nan")
        if "best_baseline_profile_shape_mae_pct" not in results.columns:
            results["best_baseline_profile_shape_mae_pct"] = float("nan")
        if "beats_persistence_phase" not in results.columns:
            results["beats_persistence_phase"] = results.get("beats_persistence_path", False)
        if "beats_persistence_next_lock" not in results.columns:
            results["beats_persistence_next_lock"] = results.get("beats_persistence_path", False)
        if "beats_persistence_profile_shape" not in results.columns:
            results["beats_persistence_profile_shape"] = False
        if "beats_best_baseline_phase" not in results.columns:
            results["beats_best_baseline_phase"] = results.get("beats_best_baseline_path", False)
        if "beats_best_baseline_next_lock" not in results.columns:
            results["beats_best_baseline_next_lock"] = results.get("beats_best_baseline_path", False)
        if "beats_best_baseline_profile_shape" not in results.columns:
            results["beats_best_baseline_profile_shape"] = False

        recommended_run_id = str(payload.get("recommended_run_id", ""))
        recommended_candidate_label = str(payload.get("recommended_candidate_label", ""))
        recommended_origin_policy = str(payload.get("recommended_origin_policy", ""))
        metric_name = str(payload.get("selection_target", "path_mae"))
        fields = _selection_metric_fields(metric_name)
        metric_column = fields["metric"].replace("learned_", "")
        metric_pct_column = f"{metric_column}_pct"
        secondary_metric = fields["secondary"].replace("learned_", "")

        matched = results.loc[results["run_id"].astype("string").eq(recommended_run_id)].copy()
        if "candidate_label" in matched.columns:
            matched = matched.loc[
                matched["candidate_label"].astype("string").eq(recommended_candidate_label)
            ].copy()
        if "requested_origin_policy" in matched.columns and recommended_origin_policy:
            matched = matched.loc[
                matched["requested_origin_policy"].astype("string").eq(recommended_origin_policy)
            ].copy()
        if matched.empty:
            matched = results.loc[
                results.get("candidate_label", pd.Series(index=results.index, dtype="string"))
                .astype("string")
                .eq(recommended_candidate_label)
            ].copy()
            if "requested_origin_policy" in matched.columns and recommended_origin_policy:
                matched = matched.loc[
                    matched["requested_origin_policy"].astype("string").eq(recommended_origin_policy)
                ].copy()
        if matched.empty:
            logger.warning(
                "Skipping challenger sweep %s because recommended candidate %s/%s was not found in %s",
                sweep_dir.name,
                recommended_run_id,
                recommended_candidate_label,
                candidate_results_path,
            )
            continue

        matched = matched.sort_values(
            [metric_column, secondary_metric, "candidate_rank"],
            ascending=[True, True, True],
            kind="stable",
        )
        row = matched.iloc[0]
        bool_payload = _coerce_bool_series(
            pd.Series(
                [
                    row.get("beats_persistence_endpoint", False),
                    row.get("beats_persistence_path", False),
                    row.get("beats_persistence_phase", False),
                    row.get("beats_persistence_next_lock", False),
                    row.get("beats_persistence_profile_shape", False),
                    row.get("beats_best_baseline_endpoint", False),
                    row.get("beats_best_baseline_path", False),
                    row.get("beats_best_baseline_phase", False),
                    row.get("beats_best_baseline_next_lock", False),
                    row.get("beats_best_baseline_profile_shape", False),
                ]
            )
        ).tolist()
        rows.append(
            {
                "sweep_run_id": sweep_dir.name,
                "generated_at_utc": payload.get("generated_at_utc"),
                "load_type": payload.get("load_type", DATASET["load_type"]),
                "artifact_namespace": payload.get("artifact_namespace", DATASET["artifact_namespace"]),
                "requested_horizon_minutes": int(payload["requested_horizon_minutes"]),
                "selection_target": metric_name,
                "origin_selection_scope": str(manifest.get("origin_selection_scope", "")),
                "shared_origin_count": int(manifest.get("shared_origin_count", row.get("origin_n", 0))),
                "recommended_origin_policy": (
                    recommended_origin_policy
                    or str(row.get("requested_origin_policy", ""))
                ),
                "recommended_candidate_label": recommended_candidate_label,
                "recommended_resolution": payload.get("recommended_resolution", row.get("resolution")),
                "recommended_feature_set": payload.get("recommended_feature_set", row.get("feature_set")),
                "recommended_model_label": payload.get("recommended_model_label", row.get("model_label")),
                "recommended_target_mode": row.get(
                    "learned_target_mode",
                    row.get("target_mode", ""),
                ),
                "recommended_source_type": payload.get("recommended_source_type", row.get("source_type", "")),
                "recommended_run_id": recommended_run_id,
                "recommended_run_path": payload.get("recommended_run_path", row.get("run_path")),
                "recommended_metric_value": float(payload.get("recommended_metric_value", row[metric_column])),
                "recommended_metric_pct": float(
                    payload.get(
                        "recommended_metric_pct",
                        row[metric_pct_column],
                    )
                ),
                "endpoint_mae": float(row["endpoint_mae"]),
                "endpoint_mae_pct": float(row["endpoint_mae_pct"]),
                "path_mae": float(row["path_mae"]),
                "path_mae_pct": float(row["path_mae_pct"]),
                "phase_mean_mae": float(row.get("phase_mean_mae", float("nan"))),
                "phase_mean_mae_pct": float(row.get("phase_mean_mae_pct", float("nan"))),
                "next_lock_mae": float(row.get("next_lock_mae", float("nan"))),
                "next_lock_mae_pct": float(row.get("next_lock_mae_pct", float("nan"))),
                "profile_shape_mae": float(row.get("profile_shape_mae", float("nan"))),
                "profile_shape_mae_pct": float(row.get("profile_shape_mae_pct", float("nan"))),
                "energy_mae": float(row.get("energy_mae", float("nan"))),
                "energy_mae_pct": float(row.get("energy_mae_pct", float("nan"))),
                "mean_coverage": float(row["mean_coverage"]),
                "origin_n": int(row["origin_n"]),
                "persistence_endpoint_mae": float(row["persistence_endpoint_mae"]),
                "persistence_endpoint_mae_pct": float(row.get("persistence_endpoint_mae_pct", float("nan"))),
                "persistence_path_mae": float(row["persistence_path_mae"]),
                "persistence_path_mae_pct": float(row.get("persistence_path_mae_pct", float("nan"))),
                "persistence_phase_mean_mae": float(
                    row.get("persistence_phase_mean_mae", float("nan"))
                ),
                "persistence_phase_mean_mae_pct": float(
                    row.get("persistence_phase_mean_mae_pct", float("nan"))
                ),
                "persistence_next_lock_mae": float(row.get("persistence_next_lock_mae", float("nan"))),
                "persistence_next_lock_mae_pct": float(
                    row.get("persistence_next_lock_mae_pct", float("nan"))
                ),
                "persistence_profile_shape_mae": float(
                    row.get("persistence_profile_shape_mae", float("nan"))
                ),
                "persistence_profile_shape_mae_pct": float(
                    row.get("persistence_profile_shape_mae_pct", float("nan"))
                ),
                "best_baseline_endpoint_label": str(row.get("best_baseline_endpoint_label", "")),
                "best_baseline_endpoint_mae": float(row.get("best_baseline_endpoint_mae", float("nan"))),
                "best_baseline_endpoint_mae_pct": float(
                    row.get("best_baseline_endpoint_mae_pct", float("nan"))
                ),
                "best_baseline_path_label": str(row.get("best_baseline_path_label", "")),
                "best_baseline_path_mae": float(row.get("best_baseline_path_mae", float("nan"))),
                "best_baseline_path_mae_pct": float(row.get("best_baseline_path_mae_pct", float("nan"))),
                "best_baseline_phase_label": str(row.get("best_baseline_phase_label", "")),
                "best_baseline_phase_mae": float(row.get("best_baseline_phase_mae", float("nan"))),
                "best_baseline_phase_mae_pct": float(
                    row.get("best_baseline_phase_mae_pct", float("nan"))
                ),
                "best_baseline_next_lock_label": str(row.get("best_baseline_next_lock_label", "")),
                "best_baseline_next_lock_mae": float(row.get("best_baseline_next_lock_mae", float("nan"))),
                "best_baseline_next_lock_mae_pct": float(
                    row.get("best_baseline_next_lock_mae_pct", float("nan"))
                ),
                "best_baseline_profile_shape_label": str(
                    row.get("best_baseline_profile_shape_label", "")
                ),
                "best_baseline_profile_shape_mae": float(
                    row.get("best_baseline_profile_shape_mae", float("nan"))
                ),
                "best_baseline_profile_shape_mae_pct": float(
                    row.get("best_baseline_profile_shape_mae_pct", float("nan"))
                ),
                "beats_persistence_endpoint": bool_payload[0],
                "beats_persistence_path": bool_payload[1],
                "beats_persistence_phase": bool_payload[2],
                "beats_persistence_next_lock": bool_payload[3],
                "beats_persistence_profile_shape": bool_payload[4],
                "beats_best_baseline_endpoint": bool_payload[5],
                "beats_best_baseline_path": bool_payload[6],
                "beats_best_baseline_phase": bool_payload[7],
                "beats_best_baseline_next_lock": bool_payload[8],
                "beats_best_baseline_profile_shape": bool_payload[9],
                "candidate_results_path": _relative_artifact_path(candidate_results_path),
                "recommended_candidate_path": _relative_artifact_path(recommended_path),
                "sweep_path": _relative_artifact_path(sweep_dir),
            }
        )

    registry = pd.DataFrame(rows, columns=CHALLENGER_SWEEP_REGISTRY_COLUMNS)
    if registry.empty:
        return registry
    registry["generated_at_utc"] = pd.to_datetime(registry["generated_at_utc"], errors="coerce", utc=True)
    for column in (
        "requested_horizon_minutes",
        "origin_n",
    ):
        registry[column] = pd.to_numeric(registry[column], errors="coerce")
    for column in (
        "recommended_metric_value",
        "recommended_metric_pct",
        "endpoint_mae",
        "endpoint_mae_pct",
        "path_mae",
        "path_mae_pct",
        "phase_mean_mae",
        "phase_mean_mae_pct",
        "next_lock_mae",
        "next_lock_mae_pct",
        "profile_shape_mae",
        "profile_shape_mae_pct",
        "energy_mae",
        "energy_mae_pct",
        "mean_coverage",
        "persistence_endpoint_mae",
        "persistence_endpoint_mae_pct",
        "persistence_path_mae",
        "persistence_path_mae_pct",
        "persistence_phase_mean_mae",
        "persistence_phase_mean_mae_pct",
        "persistence_next_lock_mae",
        "persistence_next_lock_mae_pct",
        "persistence_profile_shape_mae",
        "persistence_profile_shape_mae_pct",
        "best_baseline_endpoint_mae",
        "best_baseline_endpoint_mae_pct",
        "best_baseline_path_mae",
        "best_baseline_path_mae_pct",
        "best_baseline_phase_mae",
        "best_baseline_phase_mae_pct",
        "best_baseline_next_lock_mae",
        "best_baseline_next_lock_mae_pct",
        "best_baseline_profile_shape_mae",
        "best_baseline_profile_shape_mae_pct",
    ):
        registry[column] = pd.to_numeric(registry[column], errors="coerce")
    for column in (
        "beats_persistence_endpoint",
        "beats_persistence_path",
        "beats_persistence_phase",
        "beats_persistence_next_lock",
        "beats_persistence_profile_shape",
        "beats_best_baseline_endpoint",
        "beats_best_baseline_path",
        "beats_best_baseline_phase",
        "beats_best_baseline_next_lock",
        "beats_best_baseline_profile_shape",
    ):
        registry[column] = _coerce_bool_series(
            registry.get(column, pd.Series(False, index=registry.index))
        )
    registry["selection_target"] = registry["selection_target"].astype("string").fillna("")
    registry["recommended_origin_policy"] = (
        registry["recommended_origin_policy"].astype("string").fillna("")
    )
    registry = registry.sort_values(
        ["requested_horizon_minutes", "selection_target", "generated_at_utc", "sweep_run_id"],
        ascending=[True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    return registry


def _select_challenger_sweep_registry_candidate(
    registry: pd.DataFrame,
    *,
    requested_horizon_minutes: int,
    requested_origin_policy: str,
    selection_target: str,
) -> pd.Series | None:
    """Pick the strongest measured sweep for one horizon/objective/origin-policy request."""
    if registry.empty:
        return None
    ranked = registry.loc[
        registry["requested_horizon_minutes"].eq(int(requested_horizon_minutes))
        & registry["selection_target"].astype("string").eq(str(selection_target))
    ].copy()
    if ranked.empty:
        return None
    fields = _selection_metric_fields(str(selection_target))
    if "recommended_metric_value" not in ranked.columns:
        ranked["recommended_metric_value"] = ranked[fields["metric"].replace("learned_", "")]
    if "origin_n" not in ranked.columns:
        ranked["origin_n"] = 0
    if "mean_coverage" not in ranked.columns:
        ranked["mean_coverage"] = 0.0
    if "origin_selection_scope" not in ranked.columns:
        ranked["origin_selection_scope"] = ""
    if "shared_origin_count" not in ranked.columns:
        ranked["shared_origin_count"] = ranked["origin_n"]
    if "generated_at_utc" not in ranked.columns:
        ranked["generated_at_utc"] = ""
    ranked["origin_policy_match"] = ranked["recommended_origin_policy"].astype("string").eq(
        str(requested_origin_policy)
    )
    ranked["shared_origin_scope_match"] = ranked["origin_selection_scope"].astype("string").eq(
        "shared_timestamp_intersection"
    )
    beat_baseline = fields["beat_baseline"]
    beat_persistence = fields["beat_persistence"]
    secondary_metric = fields["secondary"].replace("learned_", "")
    if secondary_metric not in ranked.columns:
        ranked[secondary_metric] = float("nan")
    ranked = ranked.sort_values(
        [
            "origin_policy_match",
            "shared_origin_scope_match",
            beat_baseline,
            beat_persistence,
            "shared_origin_count",
            "origin_n",
            "mean_coverage",
            "recommended_metric_value",
            secondary_metric,
            "generated_at_utc",
            "sweep_run_id",
        ],
        ascending=[False, False, False, False, False, False, False, True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    return ranked.iloc[0]


def _load_stage6_registry() -> pd.DataFrame:
    """Load Stage-6 winner registry from scoped and legacy roots."""
    frames: list[pd.DataFrame] = []
    seen_roots: set[Path] = set()
    for output_root in output_path_candidates(PATHS["outputs_multires_dir"]):
        if output_root in seen_roots:
            continue
        seen_roots.add(output_root)
        registry_path = output_root / "winner_registry.csv"
        if registry_path.exists():
            frame = _read_csv_if_present(registry_path)
        elif output_root.exists():
            frame = _build_winner_registry_snapshot(output_root)
        else:
            frame = pd.DataFrame()
        if frame.empty:
            continue
        frame = frame.copy()
        frame["registry_path"] = _relative_artifact_path(registry_path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "learned_phase_mean_mae" not in combined.columns:
        combined["learned_phase_mean_mae"] = combined.get("learned_path_mae", float("nan"))
    if "learned_phase_mean_mae_pct" not in combined.columns:
        combined["learned_phase_mean_mae_pct"] = combined.get("learned_path_mae_pct", float("nan"))
    if "beats_persistence_phase" not in combined.columns:
        combined["beats_persistence_phase"] = combined.get("beats_persistence_path", False)
    if "beats_best_baseline_phase" not in combined.columns:
        combined["beats_best_baseline_phase"] = combined.get("beats_best_baseline_path", False)
    combined["generated_at_utc"] = pd.to_datetime(
        combined.get("generated_at_utc"),
        errors="coerce",
        utc=True,
    )
    combined["winner_horizon_minutes"] = pd.to_numeric(
        combined.get("winner_horizon_minutes"),
        errors="coerce",
    )
    for column in ("practical_gain_passed", "pareto_passed"):
        combined[column] = _coerce_bool_series(
            combined.get(column, pd.Series(False, index=combined.index))
        )
    combined = combined.drop_duplicates(
        subset=[
            "run_id",
            "winner_resolution",
            "winner_feature_set",
            "winner_model_label",
            "winner_forecast_strategy",
            "winner_horizon_minutes",
        ],
        keep="first",
    ).reset_index(drop=True)
    return combined


def _load_stage7_registry(output_root: Path) -> pd.DataFrame:
    """Load Stage-7 rollout registry from scoped and legacy roots."""
    frames: list[pd.DataFrame] = []
    seen_roots: set[Path] = set()
    for candidate_root in output_path_candidates(PATHS["outputs_rollout_dir"]):
        if candidate_root in seen_roots:
            continue
        seen_roots.add(candidate_root)
        registry_path = candidate_root / "rollout_registry.csv"
        if registry_path.exists():
            frame = _read_csv_if_present(registry_path)
        elif candidate_root.exists():
            frame = _build_rollout_registry_snapshot(candidate_root)
        else:
            frame = pd.DataFrame()
        if frame.empty:
            continue
        frame = frame.copy()
        frame["registry_path"] = _relative_artifact_path(registry_path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["generated_at_utc"] = pd.to_datetime(
        combined.get("generated_at_utc"),
        errors="coerce",
        utc=True,
    )
    for column in (
        "horizon_minutes",
        "origins_per_run",
        "learned_origin_n",
    ):
        combined[column] = pd.to_numeric(combined.get(column), errors="coerce")
    for column in (
        "learned_endpoint_mae",
        "learned_endpoint_mae_pct",
        "learned_path_mae",
        "learned_path_mae_pct",
        "learned_phase_mean_mae",
        "learned_phase_mean_mae_pct",
        "learned_next_lock_mae",
        "learned_next_lock_mae_pct",
        "learned_profile_shape_mae",
        "learned_profile_shape_mae_pct",
        "learned_energy_mae",
        "learned_energy_mae_pct",
        "learned_mean_coverage",
    ):
        combined[column] = pd.to_numeric(combined.get(column), errors="coerce")
    for column in (
        "beats_persistence_endpoint",
        "beats_persistence_path",
        "beats_persistence_phase",
        "beats_persistence_next_lock",
        "beats_persistence_profile_shape",
        "beats_best_baseline_endpoint",
        "beats_best_baseline_path",
        "beats_best_baseline_phase",
        "beats_best_baseline_next_lock",
        "beats_best_baseline_profile_shape",
    ):
        combined[column] = _coerce_bool_series(
            combined.get(column, pd.Series(False, index=combined.index))
        )
    combined = combined.drop_duplicates(
        subset=["run_id", "resolution", "feature_set", "model_label", "horizon_minutes"],
        keep="first",
    ).reset_index(drop=True)
    return combined


def _build_rollout_registry_candidates(
    *,
    requested_horizon_minutes: int,
    requested_origin_policy: str,
    selection_target: str,
    output_root: Path,
) -> list[dict[str, Any]]:
    """Seed challenger candidates from comparable prior Stage-7 rollout runs."""
    registry = _load_stage7_registry(output_root)
    if registry.empty:
        return []
    registry = registry.loc[registry["horizon_minutes"].eq(int(requested_horizon_minutes))].copy()
    if registry.empty:
        return []
    fields = _selection_metric_fields(str(selection_target))
    metric_column = fields["metric"]
    metric_pct_column = f"{metric_column}_pct"
    secondary_metric = fields["secondary"]
    beat_baseline_column = fields["beat_baseline"]
    beat_persistence_column = fields["beat_persistence"]
    registry["origin_policy_match"] = registry["origin_policy"].astype("string").eq(
        str(requested_origin_policy)
    )
    registry["selection_target_match"] = registry["selection_target"].astype("string").eq(
        str(selection_target)
    )
    registry = registry.sort_values(
        [
            "origin_policy_match",
            beat_baseline_column,
            beat_persistence_column,
            "selection_target_match",
            "learned_origin_n",
            "learned_mean_coverage",
            metric_column,
            secondary_metric,
            "generated_at_utc",
            "run_id",
        ],
        ascending=[False, False, False, False, False, False, True, True, False, False],
        kind="stable",
    )
    rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(registry.iterrows(), start=1):
        rows.append(
            {
                "source_priority": 1,
                "source_stage": "007_rollout",
                "source_type": "rollout_registry",
                "source_rank": rank,
                "source_run_id": row.get("run_id"),
                "source_selection_target": row.get("selection_target"),
                "source_horizon_minutes": int(row["horizon_minutes"]),
                "source_metric_value": row.get(metric_column),
                "source_metric_pct": row.get(metric_pct_column),
                "source_metric_name": selection_target,
                "source_path": row.get("registry_path"),
                "resolution": str(row["resolution"]),
                "feature_set": str(row["feature_set"]),
                "model_label": str(row["model_label"]),
                "reason": (
                    "Prior Stage-7 learned rollout candidate measured at the requested horizon; "
                    "ranked by objective-aligned registry evidence."
                ),
            }
        )
    return rows


def _build_stage6_registry_candidates(*, requested_horizon_minutes: int) -> list[dict[str, Any]]:
    """Seed challenger candidates from the Stage-6 winner registry."""
    registry = _load_stage6_registry()
    if registry.empty:
        return []
    learned = registry.loc[registry["winner_type"].astype("string").eq("learned_model")].copy()
    if learned.empty:
        return []
    if "winner_forecast_strategy" in learned.columns:
        learned = learned.loc[
            learned["winner_forecast_strategy"].astype("string").fillna("recursive").eq("recursive")
        ].copy()
    if learned.empty:
        return []
    learned["exact_horizon_match"] = learned["winner_horizon_minutes"].eq(int(requested_horizon_minutes))
    learned = learned.sort_values(
        [
            "exact_horizon_match",
            "practical_gain_passed",
            "pareto_passed",
            "winner_horizon_minutes",
            "generated_at_utc",
            "run_id",
        ],
        ascending=[False, False, False, False, False, False],
        kind="stable",
    )
    rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(learned.iterrows(), start=1):
        rows.append(
            {
                "source_priority": 2,
                "source_stage": "006_multires",
                "source_type": "winner_registry",
                "source_rank": rank,
                "source_run_id": row.get("run_id"),
                "source_selection_target": pd.NA,
                "source_horizon_minutes": int(row["winner_horizon_minutes"]),
                "source_metric_value": pd.NA,
                "source_metric_pct": pd.NA,
                "source_metric_name": "matched_horizon_selection",
                "source_path": row.get("registry_path"),
                "resolution": str(row["winner_resolution"]),
                "feature_set": str(row["winner_feature_set"]),
                "model_label": str(row["winner_model_label"]),
                "reason": (
                    "Stage-6 recursive learned winner promoted as a rollout challenger; "
                    "ranked by practical gain, Pareto pass, and longest matched horizon."
                ),
            }
        )
    return rows


def _build_horizon_policy_candidates(*, requested_horizon_minutes: int) -> list[dict[str, Any]]:
    """Seed rollout challengers from the centralized horizon policy surface."""
    policy = resolve_horizon_policy(int(requested_horizon_minutes))
    available_model_labels = set(build_model_catalog())
    target_native_steps = 18
    rows: list[dict[str, Any]] = []
    source_rank = 0
    for resolution in MULTIRES_ROLLOUT_CHALLENGERS["policy_resolutions"]:
        try:
            native_steps = lead_steps_for_horizon(str(resolution), int(requested_horizon_minutes))
        except ValueError:
            continue
        step_penalty = abs(int(native_steps) - target_native_steps)
        for feature_idx, feature_set in enumerate(policy["feature_sets"], start=1):
            for model_idx, model_label in enumerate(policy["model_labels"], start=1):
                if str(model_label) not in available_model_labels:
                    continue
                source_rank += 1
                heuristic_rank = step_penalty * 1000 + feature_idx * 100 + model_idx
                rows.append(
                    {
                        "source_priority": 3,
                        "source_stage": "policy_grid",
                        "source_type": "horizon_policy",
                        "source_rank": heuristic_rank,
                        "source_run_id": pd.NA,
                        "source_selection_target": pd.NA,
                        "source_horizon_minutes": int(requested_horizon_minutes),
                        "source_metric_value": pd.NA,
                        "source_metric_pct": pd.NA,
                        "source_metric_name": "policy_seed",
                        "source_path": "config/modeling.toml",
                        "resolution": str(resolution),
                        "feature_set": str(feature_set),
                        "model_label": str(model_label),
                        "reason": (
                            "Centralized horizon-policy challenger seeded from config/modeling.toml; "
                            "ranked by native-step suitability plus configured feature/model order."
                        ),
                    }
                )
    return rows


def _build_config_default_candidate() -> dict[str, Any]:
    """Return the static config fallback retained as the last challenger seed."""
    default_selection_target = resolve_rollout_selection_target(
        int(MULTIRES_ROLLOUT["horizon_minutes"]),
        str(MULTIRES_ROLLOUT["selection_target"]),
    )
    return {
        "source_priority": 4,
        "source_stage": "multires.toml",
        "source_type": "config_default",
        "source_rank": 1,
        "source_run_id": pd.NA,
        "source_selection_target": default_selection_target,
        "source_horizon_minutes": int(MULTIRES_ROLLOUT["horizon_minutes"]),
        "source_metric_value": pd.NA,
        "source_metric_pct": pd.NA,
        "source_metric_name": default_selection_target,
        "source_path": "config/multires.toml",
        "resolution": str(MULTIRES_ROLLOUT["selected_resolution"]),
        "feature_set": str(MULTIRES_ROLLOUT["feature_set"]),
        "model_label": str(MULTIRES_ROLLOUT["model_label"]),
        "reason": "Configured Stage-7 default retained as an operational fallback challenger.",
    }


def _requested_origin_policies_for_sweep(origin_policy: str) -> list[str]:
    """Return the origin policies that should be evaluated for one sweep request."""
    return [str(origin_policy)]


def _finalize_candidate_plan(
    candidates: list[dict[str, Any]],
    *,
    max_candidates: int,
    preserve_default: bool,
) -> pd.DataFrame:
    """Deduplicate and truncate the challenger plan while preserving the config fallback."""
    plan = pd.DataFrame(candidates)
    if plan.empty:
        return plan
    plan["candidate_key"] = (
        plan["resolution"].astype("string")
        + "|"
        + plan["feature_set"].astype("string")
        + "|"
        + plan["model_label"].astype("string")
    )
    plan = plan.sort_values(
        ["source_priority", "source_rank", "source_horizon_minutes", "candidate_key"],
        ascending=[True, True, False, True],
        kind="stable",
    ).drop_duplicates(subset=["candidate_key"], keep="first")
    default_mask = plan["source_type"].astype("string").eq("config_default")
    default_key = None
    if preserve_default and default_mask.any():
        default_key = str(plan.loc[default_mask].iloc[0]["candidate_key"])
    if len(plan) > max_candidates:
        head = plan.head(max_candidates).copy()
        if default_key is not None and not head["candidate_key"].astype("string").eq(default_key).any():
            default_row = plan.loc[plan["candidate_key"].astype("string").eq(default_key)].iloc[[0]]
            head = pd.concat([head.iloc[:-1], default_row], ignore_index=True)
        plan = head
    plan = plan.reset_index(drop=True)
    plan.insert(0, "candidate_rank", range(1, len(plan) + 1))
    return plan.drop(columns=["candidate_key"])


def _partition_representable_candidates(
    plan: pd.DataFrame,
    *,
    requested_horizon_minutes: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Keep only candidates that can express the requested horizon at native cadence."""
    if plan.empty:
        return plan, []
    runnable_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in plan.to_dict(orient="records"):
        resolution = str(row["resolution"])
        try:
            lead_steps_for_horizon(resolution, int(requested_horizon_minutes))
        except ValueError:
            warnings.append(
                "skipped_non_representable_candidate:"
                f"{resolution}:{int(requested_horizon_minutes)}:{row['feature_set']}:{row['model_label']}"
            )
            continue
        runnable_rows.append(row)
    runnable = pd.DataFrame(runnable_rows)
    if runnable.empty:
        return runnable, warnings
    runnable = runnable.reset_index(drop=True)
    runnable["candidate_rank"] = range(1, len(runnable) + 1)
    return runnable, warnings


def _build_shared_origin_timestamps(
    *,
    candidate_plan: pd.DataFrame,
    requested_horizon_minutes: int,
    origins: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a shared timestamp sample so cross-resolution challenger runs are comparable."""
    if candidate_plan.empty:
        return pd.DataFrame(columns=["requested_origin_policy", "origin_timestamp"]), []

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    requested_policy_series = (
        candidate_plan.get("requested_origin_policy", pd.Series(dtype="string"))
        .astype("string")
        .dropna()
        .drop_duplicates()
    )
    for requested_origin_policy in requested_policy_series.tolist():
        policy_plan = candidate_plan.loc[
            candidate_plan["requested_origin_policy"].astype("string").eq(str(requested_origin_policy))
        ].copy()
        if policy_plan.empty:
            continue
        shared_timestamps: set[pd.Timestamp] | None = None
        for resolution in sorted(policy_plan["resolution"].astype("string").dropna().unique().tolist()):
            base = load_base_gold(str(resolution))
            horizon_steps = lead_steps_for_horizon(str(resolution), int(requested_horizon_minutes))
            eligible_positions = _select_rollout_origins(
                base,
                horizon_steps=horizon_steps,
                max_origins=len(base),
                origin_policy=str(requested_origin_policy),
            )
            eligible_timestamps = {
                pd.Timestamp(base.iloc[position]["timestamp"])
                for position in eligible_positions
            }
            if not eligible_timestamps:
                warnings.append(
                    "no_eligible_origins_for_candidate_group:"
                    f"{requested_origin_policy}:{resolution}:{int(requested_horizon_minutes)}"
                )
                shared_timestamps = set()
                break
            if shared_timestamps is None:
                shared_timestamps = eligible_timestamps
            else:
                shared_timestamps &= eligible_timestamps
            if not shared_timestamps:
                break
        if not shared_timestamps:
            resolutions = ",".join(sorted(policy_plan["resolution"].astype("string").unique().tolist()))
            warnings.append(
                "no_shared_origins_across_candidates:"
                f"{requested_origin_policy}:{int(requested_horizon_minutes)}:{resolutions}"
            )
            continue
        sampled = _sample_rollout_origin_timestamps(
            list(shared_timestamps),
            max_origins=int(origins),
            origin_policy=str(requested_origin_policy),
        )
        if len(sampled) < int(origins):
            warnings.append(
                "shared_origin_count_limited:"
                f"{requested_origin_policy}:{len(sampled)}:{int(origins)}"
            )
        rows.extend(
            {
                "requested_origin_policy": str(requested_origin_policy),
                "origin_timestamp": pd.Timestamp(timestamp).isoformat(),
            }
            for timestamp in sampled
        )
    return pd.DataFrame(rows), warnings


def _build_challenger_plan(
    *,
    requested_horizon_minutes: int,
    requested_origin_policy: str,
    selection_target: str,
    output_root: Path,
    max_candidates: int,
) -> pd.DataFrame:
    """Build the ordered challenger plan from registries, policy seeds, and config fallback."""
    candidates: list[dict[str, Any]] = []
    if MULTIRES_ROLLOUT_CHALLENGERS["include_rollout_registry"]:
        candidates.extend(
            _build_rollout_registry_candidates(
                requested_horizon_minutes=requested_horizon_minutes,
                requested_origin_policy=requested_origin_policy,
                selection_target=selection_target,
                output_root=output_root,
            )
        )
    if MULTIRES_ROLLOUT_CHALLENGERS["include_stage6_registry"]:
        candidates.extend(
            _build_stage6_registry_candidates(requested_horizon_minutes=requested_horizon_minutes)
        )
    if MULTIRES_ROLLOUT_CHALLENGERS["include_horizon_policy_candidates"]:
        candidates.extend(
            _build_horizon_policy_candidates(requested_horizon_minutes=requested_horizon_minutes)
        )
    if MULTIRES_ROLLOUT_CHALLENGERS["include_config_default"]:
        candidates.append(_build_config_default_candidate())
    return _finalize_candidate_plan(
        candidates,
        max_candidates=max_candidates,
        preserve_default=bool(MULTIRES_ROLLOUT_CHALLENGERS["include_config_default"]),
    )


def _selection_context_from_plan_row(
    row: pd.Series,
    *,
    requested_horizon_minutes: int,
    requested_origin_policy: str,
    selection_target: str,
) -> dict[str, Any]:
    """Translate one challenger-plan row into a normal Stage-7 selection payload."""
    source_stage = str(row["source_stage"])
    source_horizon = int(row["source_horizon_minutes"])
    return {
        "resolution": str(row["resolution"]),
        "feature_set": str(row["feature_set"]),
        "model_label": str(row["model_label"]),
        "forecast_strategy": "recursive",
        "requested_horizon_minutes": int(requested_horizon_minutes),
        "requested_origin_policy": str(requested_origin_policy),
        "selection_target": str(selection_target),
        "matched_stage6_horizon_minutes": source_horizon if source_stage == "006_multires" else None,
        "matched_rollout_registry_horizon_minutes": (
            source_horizon if source_stage == "007_rollout" else None
        ),
        "selection_source": str(row["source_path"]),
        "selection_policy": "challenger_sweep_candidate",
        "selection_reason": str(row["reason"]),
        "selection_run_id": None if pd.isna(row.get("source_run_id")) else str(row.get("source_run_id")),
        "selection_run_stage": None if source_stage == "multires.toml" else source_stage,
        "explicit_candidate_override": True,
    }


def _candidate_run_summary(
    *,
    result: dict[str, Any],
    plan_row: pd.Series,
    selection_target: str,
) -> dict[str, Any]:
    """Summarize one challenger rollout run into the sweep comparison schema."""
    metrics = cast(pd.DataFrame, result["metrics"]).copy()
    if "phase_mean_mae" not in metrics.columns:
        metrics["phase_mean_mae"] = metrics.get("path_mae", float("nan"))
    if "phase_mean_mae_pct" not in metrics.columns:
        metrics["phase_mean_mae_pct"] = metrics.get("path_mae_pct", float("nan"))
    if "next_lock_mae" not in metrics.columns:
        metrics["next_lock_mae"] = metrics.get("path_mae", float("nan"))
    if "next_lock_mae_pct" not in metrics.columns:
        metrics["next_lock_mae_pct"] = metrics.get("path_mae_pct", float("nan"))
    if "profile_shape_mae" not in metrics.columns:
        metrics["profile_shape_mae"] = float("nan")
    if "profile_shape_mae_pct" not in metrics.columns:
        metrics["profile_shape_mae_pct"] = float("nan")
    if "energy_mae" not in metrics.columns:
        metrics["energy_mae"] = float("nan")
    if "energy_mae_pct" not in metrics.columns:
        metrics["energy_mae_pct"] = float("nan")
    candidate_row = metrics.loc[
        metrics.get("candidate_type", pd.Series(index=metrics.index, dtype="string"))
        .astype("string")
        .fillna("baseline")
        .eq("learned")
    ]
    if candidate_row.empty:
        candidate_row = metrics.loc[
            metrics["candidate_label"].astype("string").str.startswith(str(plan_row["model_label"]))
        ]
    if candidate_row.empty:
        raise RuntimeError(
            f"Missing learned rollout metrics for candidate={plan_row['model_label']} in {result['run_dir']}"
        )
    fields = _selection_metric_fields(str(selection_target))
    metric_name = fields["metric"].replace("learned_", "")
    metric_pct_name = f"{metric_name}_pct"
    secondary_metric_name = fields["secondary"].replace("learned_", "")
    candidate = candidate_row.sort_values(
        [metric_name, secondary_metric_name, "candidate_label"],
        ascending=[True, True, True],
        kind="stable",
    ).iloc[0]
    persistence = metrics.loc[metrics["candidate_label"].astype("string").eq("persistence")]
    baselines = metrics.loc[
        metrics.get("candidate_type", pd.Series(index=metrics.index, dtype="string"))
        .astype("string")
        .fillna("baseline")
        .eq("baseline")
    ].copy()
    best_baseline_endpoint = pd.Series(dtype=object)
    best_baseline_path = pd.Series(dtype=object)
    best_baseline_phase = pd.Series(dtype=object)
    best_baseline_next_lock = pd.Series(dtype=object)
    best_baseline_profile_shape = pd.Series(dtype=object)
    if not baselines.empty:
        best_baseline_endpoint = baselines.sort_values(
            ["endpoint_mae", "path_mae", "candidate_label"],
            ascending=[True, True, True],
            kind="stable",
        ).iloc[0]
        best_baseline_path = baselines.sort_values(
            ["path_mae", "endpoint_mae", "candidate_label"],
            ascending=[True, True, True],
            kind="stable",
        ).iloc[0]
        best_baseline_phase = baselines.sort_values(
            ["phase_mean_mae", "path_mae", "candidate_label"],
            ascending=[True, True, True],
            kind="stable",
        ).iloc[0]
        best_baseline_next_lock = baselines.sort_values(
            ["next_lock_mae", "path_mae", "candidate_label"],
            ascending=[True, True, True],
            kind="stable",
        ).iloc[0]
        best_baseline_profile_shape = baselines.sort_values(
            ["profile_shape_mae", "path_mae", "candidate_label"],
            ascending=[True, True, True],
            kind="stable",
        ).iloc[0]
    return {
        "candidate_rank": int(plan_row["candidate_rank"]),
        "candidate_label": str(candidate["candidate_label"]),
        "resolution": str(plan_row["resolution"]),
        "feature_set": str(plan_row["feature_set"]),
        "model_label": str(plan_row["model_label"]),
        "learned_target_mode": str(candidate.get("target_mode", "raw")),
        "source_stage": str(plan_row["source_stage"]),
        "source_type": str(plan_row["source_type"]),
        "source_run_id": None if pd.isna(plan_row["source_run_id"]) else str(plan_row["source_run_id"]),
        "source_horizon_minutes": int(plan_row["source_horizon_minutes"]),
        "requested_origin_policy": str(plan_row["requested_origin_policy"]),
        "run_id": result["run_dir"].name,
        "run_path": _relative_artifact_path(result["run_dir"]),
        "selection_target": str(selection_target),
        "selection_metric_name": metric_name,
        "selection_metric_value": float(candidate[metric_name]),
        "selection_metric_pct": float(candidate[metric_pct_name]),
        "endpoint_mae": float(candidate["endpoint_mae"]),
        "endpoint_mae_pct": float(candidate["endpoint_mae_pct"]),
        "path_mae": float(candidate["path_mae"]),
        "path_mae_pct": float(candidate["path_mae_pct"]),
        "phase_mean_mae": float(candidate.get("phase_mean_mae", float("nan"))),
        "phase_mean_mae_pct": float(candidate.get("phase_mean_mae_pct", float("nan"))),
        "next_lock_mae": float(candidate.get("next_lock_mae", float("nan"))),
        "next_lock_mae_pct": float(candidate.get("next_lock_mae_pct", float("nan"))),
        "profile_shape_mae": float(candidate.get("profile_shape_mae", float("nan"))),
        "profile_shape_mae_pct": float(candidate.get("profile_shape_mae_pct", float("nan"))),
        "energy_mae": float(candidate.get("energy_mae", float("nan"))),
        "energy_mae_pct": float(candidate.get("energy_mae_pct", float("nan"))),
        "mean_coverage": float(candidate["mean_coverage"]),
        "origin_n": int(candidate["origin_n"]),
        "persistence_endpoint_mae": (
            float(persistence.iloc[0]["endpoint_mae"]) if not persistence.empty else float("nan")
        ),
        "persistence_endpoint_mae_pct": (
            float(persistence.iloc[0].get("endpoint_mae_pct", float("nan")))
            if not persistence.empty
            else float("nan")
        ),
        "persistence_path_mae": (
            float(persistence.iloc[0]["path_mae"]) if not persistence.empty else float("nan")
        ),
        "persistence_path_mae_pct": (
            float(persistence.iloc[0].get("path_mae_pct", float("nan")))
            if not persistence.empty
            else float("nan")
        ),
        "persistence_phase_mean_mae": (
            float(persistence.iloc[0].get("phase_mean_mae", float("nan")))
            if not persistence.empty
            else float("nan")
        ),
        "persistence_phase_mean_mae_pct": (
            float(persistence.iloc[0].get("phase_mean_mae_pct", float("nan")))
            if not persistence.empty
            else float("nan")
        ),
        "persistence_next_lock_mae": (
            float(persistence.iloc[0].get("next_lock_mae", float("nan")))
            if not persistence.empty
            else float("nan")
        ),
        "persistence_next_lock_mae_pct": (
            float(persistence.iloc[0].get("next_lock_mae_pct", float("nan")))
            if not persistence.empty
            else float("nan")
        ),
        "persistence_profile_shape_mae": (
            float(persistence.iloc[0].get("profile_shape_mae", float("nan")))
            if not persistence.empty
            else float("nan")
        ),
        "persistence_profile_shape_mae_pct": (
            float(persistence.iloc[0].get("profile_shape_mae_pct", float("nan")))
            if not persistence.empty
            else float("nan")
        ),
        "best_baseline_endpoint_label": (
            str(best_baseline_endpoint.get("candidate_label")) if not best_baseline_endpoint.empty else ""
        ),
        "best_baseline_endpoint_mae": (
            float(best_baseline_endpoint.get("endpoint_mae")) if not best_baseline_endpoint.empty else float("nan")
        ),
        "best_baseline_endpoint_mae_pct": (
            float(best_baseline_endpoint.get("endpoint_mae_pct"))
            if not best_baseline_endpoint.empty
            else float("nan")
        ),
        "best_baseline_path_label": (
            str(best_baseline_path.get("candidate_label")) if not best_baseline_path.empty else ""
        ),
        "best_baseline_path_mae": (
            float(best_baseline_path.get("path_mae")) if not best_baseline_path.empty else float("nan")
        ),
        "best_baseline_path_mae_pct": (
            float(best_baseline_path.get("path_mae_pct")) if not best_baseline_path.empty else float("nan")
        ),
        "best_baseline_phase_label": (
            str(best_baseline_phase.get("candidate_label")) if not best_baseline_phase.empty else ""
        ),
        "best_baseline_phase_mae": (
            float(best_baseline_phase.get("phase_mean_mae")) if not best_baseline_phase.empty else float("nan")
        ),
        "best_baseline_phase_mae_pct": (
            float(best_baseline_phase.get("phase_mean_mae_pct"))
            if not best_baseline_phase.empty
            else float("nan")
        ),
        "best_baseline_next_lock_label": (
            str(best_baseline_next_lock.get("candidate_label")) if not best_baseline_next_lock.empty else ""
        ),
        "best_baseline_next_lock_mae": (
            float(best_baseline_next_lock.get("next_lock_mae"))
            if not best_baseline_next_lock.empty
            else float("nan")
        ),
        "best_baseline_next_lock_mae_pct": (
            float(best_baseline_next_lock.get("next_lock_mae_pct"))
            if not best_baseline_next_lock.empty
            else float("nan")
        ),
        "best_baseline_profile_shape_label": (
            str(best_baseline_profile_shape.get("candidate_label"))
            if not best_baseline_profile_shape.empty
            else ""
        ),
        "best_baseline_profile_shape_mae": (
            float(best_baseline_profile_shape.get("profile_shape_mae"))
            if not best_baseline_profile_shape.empty
            else float("nan")
        ),
        "best_baseline_profile_shape_mae_pct": (
            float(best_baseline_profile_shape.get("profile_shape_mae_pct"))
            if not best_baseline_profile_shape.empty
            else float("nan")
        ),
        "beats_persistence_endpoint": (
            float(candidate["endpoint_mae"]) < float(persistence.iloc[0]["endpoint_mae"])
            if not persistence.empty
            else False
        ),
        "beats_persistence_path": (
            float(candidate["path_mae"]) < float(persistence.iloc[0]["path_mae"])
            if not persistence.empty
            else False
        ),
        "beats_persistence_phase": (
            float(candidate["phase_mean_mae"]) < float(persistence.iloc[0]["phase_mean_mae"])
            if not persistence.empty
            else False
        ),
        "beats_persistence_next_lock": (
            float(candidate.get("next_lock_mae", float("nan")))
            < float(persistence.iloc[0].get("next_lock_mae", float("nan")))
            if not persistence.empty
            else False
        ),
        "beats_persistence_profile_shape": (
            float(candidate.get("profile_shape_mae", float("nan")))
            < float(persistence.iloc[0].get("profile_shape_mae", float("nan")))
            if not persistence.empty
            else False
        ),
        "beats_best_baseline_endpoint": (
            float(candidate["endpoint_mae"]) < float(best_baseline_endpoint["endpoint_mae"])
            if not best_baseline_endpoint.empty
            else False
        ),
        "beats_best_baseline_path": (
            float(candidate["path_mae"]) < float(best_baseline_path["path_mae"])
            if not best_baseline_path.empty
            else False
        ),
        "beats_best_baseline_phase": (
            float(candidate["phase_mean_mae"]) < float(best_baseline_phase["phase_mean_mae"])
            if not best_baseline_phase.empty
            else False
        ),
        "beats_best_baseline_next_lock": (
            float(candidate.get("next_lock_mae", float("nan")))
            < float(best_baseline_next_lock.get("next_lock_mae", float("nan")))
            if not best_baseline_next_lock.empty
            else False
        ),
        "beats_best_baseline_profile_shape": (
            float(candidate.get("profile_shape_mae", float("nan")))
            < float(best_baseline_profile_shape.get("profile_shape_mae", float("nan")))
            if not best_baseline_profile_shape.empty
            else False
        ),
        "secondary_metric_value": float(candidate[secondary_metric_name]),
        "reason": str(plan_row["reason"]),
    }


def _sweep_policy_metric_fields(selection_target: str) -> dict[str, str]:
    """Return the by-origin metric fields used for sweep-derived policy synthesis."""
    if selection_target == "next_lock_mae":
        return {
            "metric": "next_lock_mae",
            "secondary": "path_mae",
            "policy_suffix": "next_lock",
        }
    if selection_target == "path_mae":
        return {
            "metric": "path_mae",
            "secondary": "endpoint_abs_error",
            "policy_suffix": "path",
        }
    if selection_target == "profile_shape_mae":
        return {
            "metric": "profile_shape_mae",
            "secondary": "path_mae",
            "policy_suffix": "profile_shape",
        }
    raise ValueError(f"Unsupported sweep policy selection target: {selection_target}")


def _select_sweep_policy_source_candidates(results: pd.DataFrame) -> pd.DataFrame:
    """Pick the candidate runs eligible to seed a sweep-derived portfolio policy."""
    if (
        results.empty
        or not bool(MULTIRES_ROLLOUT_SWEEP_POLICIES["enabled"])
        or int(MULTIRES_ROLLOUT_SWEEP_POLICIES["max_source_candidates"]) <= 0
    ):
        return pd.DataFrame()
    selected_rows: list[pd.Series] = []
    for selection_target in MULTIRES_ROLLOUT_SWEEP_POLICIES["selection_targets"]:
        fields = _selection_metric_fields(str(selection_target))
        metric_column = fields["metric"].replace("learned_", "")
        secondary_column = fields["secondary"].replace("learned_", "")
        if metric_column not in results.columns or secondary_column not in results.columns:
            continue
        ranked = results.sort_values(
            [metric_column, secondary_column, "candidate_rank", "candidate_label"],
            ascending=[True, True, True, True],
            kind="stable",
        ).reset_index(drop=True)
        if ranked.empty:
            continue
        winner = ranked.iloc[0].copy()
        winner["policy_source_metric_name"] = str(selection_target)
        selected_rows.append(winner)
    if not selected_rows:
        return pd.DataFrame()
    sources = pd.DataFrame(selected_rows)
    sources["candidate_key"] = (
        sources["run_id"].astype("string")
        + "|"
        + sources["candidate_label"].astype("string")
    )
    sources = sources.drop_duplicates(subset=["candidate_key"], keep="first").reset_index(drop=True)
    min_candidates = int(MULTIRES_ROLLOUT_SWEEP_POLICIES["min_source_candidates"])
    if len(sources) < min_candidates:
        return pd.DataFrame()
    max_candidates = int(MULTIRES_ROLLOUT_SWEEP_POLICIES["max_source_candidates"])
    if len(sources) > max_candidates:
        sources = sources.head(max_candidates).copy()
    return sources.drop(columns=["candidate_key"])


def _build_cross_candidate_phase_bucket_policy_by_origin(
    *,
    source_run_records: list[dict[str, Any]],
    selection_target: str,
    requested_origin_policy: str,
) -> tuple[pd.DataFrame, dict[str, Any]] | tuple[pd.DataFrame, None]:
    """Synthesize one phase-bucket portfolio policy from shared-origin sweep evidence."""
    fields = _sweep_policy_metric_fields(str(selection_target))
    source_candidate_frames: list[pd.DataFrame] = []
    source_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for record in source_run_records:
        summary = cast(dict[str, Any], record["summary"])
        result = cast(dict[str, Any], record["result"])
        run_id = str(summary["run_id"])
        candidate_label = str(summary["candidate_label"])
        by_origin = cast(pd.DataFrame, result["by_origin"]).copy()
        candidate_rows = by_origin.loc[
            by_origin["candidate_label"].astype("string").eq(candidate_label)
        ].copy()
        if candidate_rows.empty:
            continue
        candidate_rows["origin_timestamp"] = pd.to_datetime(candidate_rows["origin_timestamp"])
        candidate_rows["phase_bucket_seconds"] = candidate_rows["origin_timestamp"].map(_phase_bucket_seconds)
        candidate_rows["source_run_id"] = run_id
        candidate_rows["source_candidate_label"] = candidate_label
        candidate_rows["source_resolution"] = str(summary["resolution"])
        candidate_rows["source_feature_set"] = str(summary["feature_set"])
        candidate_rows["source_model_label"] = str(summary["model_label"])
        source_candidate_frames.append(candidate_rows)
        source_lookup[(run_id, candidate_label)] = {
            "summary": summary,
            "by_origin": by_origin.assign(origin_timestamp=pd.to_datetime(by_origin["origin_timestamp"])),
        }
    if not source_candidate_frames:
        return pd.DataFrame(), None
    working = pd.concat(source_candidate_frames, ignore_index=True)
    unique_buckets = sorted({int(value) for value in working["phase_bucket_seconds"].dropna().tolist()})
    if len(unique_buckets) < 2:
        return pd.DataFrame(), None

    bucket_mapping: dict[int, dict[str, Any]] = {}
    bucket_details: list[dict[str, Any]] = []
    for bucket in unique_buckets:
        bucket_rows = working.loc[working["phase_bucket_seconds"].eq(int(bucket))].copy()
        if bucket_rows.empty:
            continue
        ranked = (
            bucket_rows.groupby(
                [
                    "source_run_id",
                    "source_candidate_label",
                    "source_resolution",
                    "source_feature_set",
                    "source_model_label",
                ],
                dropna=False,
            )
            .agg(
                metric_mean=(fields["metric"], "mean"),
                secondary_mean=(fields["secondary"], "mean"),
                origin_n=("origin_timestamp", "nunique"),
            )
            .reset_index()
            .sort_values(
                ["metric_mean", "secondary_mean", "source_candidate_label", "source_run_id"],
                ascending=[True, True, True, True],
                kind="stable",
            )
            .reset_index(drop=True)
        )
        winner = ranked.iloc[0]
        bucket_mapping[int(bucket)] = {
            "run_id": str(winner["source_run_id"]),
            "candidate_label": str(winner["source_candidate_label"]),
            "resolution": str(winner["source_resolution"]),
            "feature_set": str(winner["source_feature_set"]),
            "model_label": str(winner["source_model_label"]),
        }
        bucket_details.append(
            {
                "phase_bucket_seconds": int(bucket),
                "selected_run_id": str(winner["source_run_id"]),
                "selected_candidate_label": str(winner["source_candidate_label"]),
                "selected_resolution": str(winner["source_resolution"]),
                "selected_feature_set": str(winner["source_feature_set"]),
                "selected_model_label": str(winner["source_model_label"]),
                "mean_metric_value": float(winner["metric_mean"]),
                "mean_secondary_value": float(winner["secondary_mean"]),
                "origin_n": int(winner["origin_n"]),
            }
        )
    if len(bucket_mapping) != len(unique_buckets):
        return pd.DataFrame(), None

    selector_label = f"cross_candidate_portfolio::phase_bucket_{fields['policy_suffix']}_policy"
    selector_target_mode = f"phase_bucket_{fields['policy_suffix']}_policy"
    unique_origins = (
        working.loc[:, ["origin_timestamp", "phase_bucket_seconds"]]
        .drop_duplicates()
        .sort_values("origin_timestamp", kind="stable")
        .reset_index(drop=True)
    )
    selector_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    origin_details: list[dict[str, Any]] = []
    for _, origin_row in unique_origins.iterrows():
        origin_timestamp = pd.Timestamp(origin_row["origin_timestamp"])
        bucket = int(origin_row["phase_bucket_seconds"])
        selected = bucket_mapping[bucket]
        source_key = (str(selected["run_id"]), str(selected["candidate_label"]))
        source_payload = source_lookup.get(source_key)
        if source_payload is None:
            continue
        source_by_origin = cast(pd.DataFrame, source_payload["by_origin"])
        matched_candidate = source_by_origin.loc[
            source_by_origin["origin_timestamp"].eq(origin_timestamp)
            & source_by_origin["candidate_label"].astype("string").eq(str(selected["candidate_label"]))
        ].copy()
        if matched_candidate.empty:
            continue
        row_payload = matched_candidate.iloc[0].to_dict()
        row_payload["candidate_label"] = selector_label
        row_payload["candidate_type"] = "learned"
        row_payload["source_model_label"] = "cross_candidate_portfolio"
        row_payload["target_mode"] = selector_target_mode
        row_payload["policy_selection_target"] = str(selection_target)
        row_payload["policy_phase_bucket_seconds"] = int(bucket)
        row_payload["policy_source_run_id"] = str(selected["run_id"])
        row_payload["policy_source_candidate"] = str(selected["candidate_label"])
        row_payload["policy_source_resolution"] = str(selected["resolution"])
        row_payload["policy_source_feature_set"] = str(selected["feature_set"])
        row_payload["policy_source_model_label"] = str(selected["model_label"])
        selector_rows.append(row_payload)

        matched_baselines = source_by_origin.loc[
            source_by_origin["origin_timestamp"].eq(origin_timestamp)
            & source_by_origin.get(
                "candidate_type",
                pd.Series(index=source_by_origin.index, dtype="string"),
            )
            .astype("string")
            .fillna("baseline")
            .eq("baseline")
        ].copy()
        if not matched_baselines.empty:
            baseline_rows.extend(matched_baselines.to_dict(orient="records"))
        origin_details.append(
            {
                "origin_timestamp": origin_timestamp.isoformat(),
                "phase_bucket_seconds": int(bucket),
                "selected_run_id": str(selected["run_id"]),
                "selected_candidate_label": str(selected["candidate_label"]),
                "selected_resolution": str(selected["resolution"]),
                "selected_feature_set": str(selected["feature_set"]),
                "selected_model_label": str(selected["model_label"]),
            }
        )
    if not selector_rows:
        return pd.DataFrame(), None
    combined = pd.DataFrame([*selector_rows, *baseline_rows])
    metadata = {
        "candidate_label": selector_label,
        "target_mode": selector_target_mode,
        "selection_target": str(selection_target),
        "requested_origin_policy": str(requested_origin_policy),
        "source_candidates": [
            {
                "run_id": str(record["summary"]["run_id"]),
                "candidate_label": str(record["summary"]["candidate_label"]),
                "resolution": str(record["summary"]["resolution"]),
                "feature_set": str(record["summary"]["feature_set"]),
                "model_label": str(record["summary"]["model_label"]),
                "policy_source_metric_name": str(record["summary"].get("policy_source_metric_name", "")),
            }
            for record in source_run_records
        ],
        "phase_bucket_mapping": bucket_mapping,
        "phase_bucket_details": bucket_details,
        "origin_details": origin_details,
    }
    return combined, metadata


def _build_cross_candidate_phase_bucket_policies(
    *,
    run_records: list[dict[str, Any]],
    candidate_results: pd.DataFrame,
    sweep_dir: Path,
    requested_horizon_minutes: int,
    requested_origin_policy: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    """Build all sweep-derived portfolio candidates permitted for this sweep."""
    if (
        candidate_results.empty
        or not bool(MULTIRES_ROLLOUT_SWEEP_POLICIES["enabled"])
        or int(requested_horizon_minutes) < int(MULTIRES_ROLLOUT_SWEEP_POLICIES["min_horizon_minutes"])
        or int(requested_horizon_minutes) > int(MULTIRES_ROLLOUT_SWEEP_POLICIES["max_horizon_minutes"])
        or str(requested_origin_policy) not in set(MULTIRES_ROLLOUT_SWEEP_POLICIES["origin_policies"])
    ):
        return pd.DataFrame(), [], pd.DataFrame()

    source_candidates = _select_sweep_policy_source_candidates(candidate_results)
    if source_candidates.empty:
        return pd.DataFrame(), [], pd.DataFrame()
    source_keys = {
        (str(row["run_id"]), str(row["candidate_label"])): row.to_dict()
        for _, row in source_candidates.iterrows()
    }
    source_run_records = [
        {
            "summary": record["summary"],
            "result": record["result"],
        }
        for record in run_records
        if (str(record["summary"]["run_id"]), str(record["summary"]["candidate_label"])) in source_keys
    ]
    if len(source_run_records) < int(MULTIRES_ROLLOUT_SWEEP_POLICIES["min_source_candidates"]):
        return pd.DataFrame(), [], pd.DataFrame()

    policy_rows: list[dict[str, Any]] = []
    policy_metadata: list[dict[str, Any]] = []
    policy_by_origin_frames: list[pd.DataFrame] = []
    base_rank = int(candidate_results["candidate_rank"].max()) if not candidate_results.empty else 0
    for idx, selection_target in enumerate(MULTIRES_ROLLOUT_SWEEP_POLICIES["selection_targets"], start=1):
        combined_by_origin, metadata = _build_cross_candidate_phase_bucket_policy_by_origin(
            source_run_records=source_run_records,
            selection_target=str(selection_target),
            requested_origin_policy=str(requested_origin_policy),
        )
        if metadata is None or combined_by_origin.empty:
            continue
        metrics = _aggregate_rollout_metrics(combined_by_origin)
        plan_row = pd.Series(
            {
                "candidate_rank": base_rank + idx,
                "resolution": "mixed",
                "feature_set": "portfolio",
                "model_label": "cross_candidate_portfolio",
                "source_stage": "007_rollout",
                "source_type": "cross_candidate_phase_bucket_portfolio",
                "source_run_id": sweep_dir.name,
                "source_horizon_minutes": int(requested_horizon_minutes),
                "requested_origin_policy": str(requested_origin_policy),
                "reason": (
                    "Cross-candidate phase-bucket portfolio synthesized from shared-origin "
                    f"rollout evidence for {selection_target}."
                ),
            }
        )
        summary = _candidate_run_summary(
            result={"run_dir": sweep_dir, "metrics": metrics},
            plan_row=plan_row,
            selection_target=str(selection_target),
        )
        policy_rows.append(summary)
        metadata["summary"] = {
            "selection_metric_name": str(summary["selection_metric_name"]),
            "selection_metric_value": float(summary["selection_metric_value"]),
            "selection_metric_pct": float(summary["selection_metric_pct"]),
            "path_mae": float(summary["path_mae"]),
            "path_mae_pct": float(summary["path_mae_pct"]),
            "next_lock_mae": float(summary["next_lock_mae"]),
            "next_lock_mae_pct": float(summary["next_lock_mae_pct"]),
            "profile_shape_mae": float(summary["profile_shape_mae"]),
            "profile_shape_mae_pct": float(summary["profile_shape_mae_pct"]),
            "endpoint_mae": float(summary["endpoint_mae"]),
            "endpoint_mae_pct": float(summary["endpoint_mae_pct"]),
            "beats_persistence_next_lock": bool(summary["beats_persistence_next_lock"]),
            "beats_best_baseline_next_lock": bool(summary["beats_best_baseline_next_lock"]),
            "beats_persistence_path": bool(summary["beats_persistence_path"]),
            "beats_best_baseline_path": bool(summary["beats_best_baseline_path"]),
            "beats_persistence_profile_shape": bool(summary["beats_persistence_profile_shape"]),
            "beats_best_baseline_profile_shape": bool(summary["beats_best_baseline_profile_shape"]),
        }
        policy_metadata.append(metadata)
        policy_frame = combined_by_origin.copy()
        policy_frame["portfolio_candidate_label"] = str(summary["candidate_label"])
        policy_frame["portfolio_selection_target"] = str(selection_target)
        policy_by_origin_frames.append(policy_frame)
    if not policy_rows:
        return pd.DataFrame(), [], pd.DataFrame()
    return (
        pd.DataFrame(policy_rows),
        policy_metadata,
        pd.concat(policy_by_origin_frames, ignore_index=True),
    )


def _select_recommended_candidate(
    results: pd.DataFrame,
    *,
    selection_target: str,
    requested_origin_policy: str,
    recommendation_origin_scope: str,
) -> pd.Series:
    """Select the operational sweep winner after applying origin and objective priorities."""
    fields = _selection_metric_fields(str(selection_target))
    metric_column = fields["metric"].replace("learned_", "")
    secondary_metric = fields["secondary"].replace("learned_", "")
    beat_baseline = fields["beat_baseline"]
    beat_persistence = fields["beat_persistence"]
    ranked = results.copy()
    if "requested_origin_policy" not in ranked.columns:
        ranked["requested_origin_policy"] = str(requested_origin_policy)
    ranked["origin_policy_match"] = ranked["requested_origin_policy"].astype("string").eq(
        str(requested_origin_policy)
    )
    origin_columns = (
        ["origin_policy_match"] if recommendation_origin_scope == "requested_only" else []
    )
    ranked = ranked.sort_values(
        [*origin_columns, beat_baseline, beat_persistence, "mean_coverage", metric_column, secondary_metric, "candidate_rank"],
        ascending=[*[False] * len(origin_columns), False, False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return ranked.iloc[0]


def _write_sweep_summary_md(
    *,
    results: pd.DataFrame,
    recommended: pd.Series,
    output_dir: Path,
    selection_target: str,
) -> None:
    """Write the human-readable markdown summary for one challenger sweep."""
    metric_label = _selection_metric_fields(str(selection_target))["label"]
    lines = [
        "# Rollout Challenger Sweep",
        "",
        f"- Objective: `{selection_target}` ({metric_label})",
        f"- Recommended candidate: `{recommended['candidate_label']}`",
        f"- Recommended run: `{recommended['run_id']}`",
        f"- Recommended {metric_label}: `{float(recommended['selection_metric_value']):.6f}`",
        f"- Recommended {metric_label} %: `{float(recommended['selection_metric_pct']):.3f}%`",
        "",
        "## Candidate Results",
        "",
    ]
    for _, row in results.iterrows():
        lines.extend(
            [
                f"- `{row['candidate_label']}` | {metric_label}=`{float(row['selection_metric_value']):.6f}` | "
                f"{metric_label} %=`{float(row['selection_metric_pct']):.3f}%` | "
                f"endpoint=`{float(row['endpoint_mae']):.6f}` ({float(row['endpoint_mae_pct']):.3f}%) | "
                f"path=`{float(row['path_mae']):.6f}` ({float(row['path_mae_pct']):.3f}%) | "
                f"phase=`{float(row['phase_mean_mae']):.6f}` ({float(row['phase_mean_mae_pct']):.3f}%) | "
                f"next-lock=`{float(row.get('next_lock_mae', float('nan'))):.6f}` "
                f"({float(row.get('next_lock_mae_pct', float('nan'))):.3f}%) | "
                f"profile-shape=`{float(row.get('profile_shape_mae', float('nan'))):.6f}` "
                f"({float(row.get('profile_shape_mae_pct', float('nan'))):.3f}%) | "
                f"source=`{row['source_type']}`"
            ]
        )
    (output_dir / "challenger_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_challenger_plan_row(
    *,
    output_root: Path,
    plan_row: dict[str, Any],
    horizon_minutes: int,
    origins: int,
    selection_target: str,
    shared_origin_map: dict[str, list[pd.Timestamp]],
) -> dict[str, Any]:
    """Execute one planned challenger rollout run and return the sweep-comparison record."""
    row = pd.Series(plan_row)
    selection = _selection_context_from_plan_row(
        row,
        requested_horizon_minutes=int(horizon_minutes),
        requested_origin_policy=str(row["requested_origin_policy"]),
        selection_target=str(selection_target),
    )
    result = run_rollout_evaluation(
        output_root=output_root,
        selection=selection,
        horizon_minutes=int(horizon_minutes),
        origins=int(origins),
        origin_policy=str(row["requested_origin_policy"]),
        selection_target=str(selection_target),
        origin_timestamps=shared_origin_map.get(str(row["requested_origin_policy"]), []),
        candidate_scope="selected_plus_baselines",
        refresh_root_registry=False,
        refresh_latest_alias=False,
    )
    summary_row = _candidate_run_summary(
        result=result,
        plan_row=row,
        selection_target=str(selection_target),
    )
    return {
        "summary": summary_row,
        "result": result,
        "plan_row": row.copy(),
    }


def run_rollout_challenger_sweep(
    *,
    output_root: Path,
    horizon_minutes: int,
    origins: int,
    origin_policy: str,
    selection_target: str,
    max_candidates: int,
    refresh_rollout_registry: bool = True,
    refresh_rollout_latest: bool = True,
    refresh_sweep_latest: bool = True,
) -> dict[str, Any]:
    """Execute one Stage-7 challenger sweep and return its core artifacts."""
    if int(max_candidates) <= 0:
        raise ValueError("max_candidates must be positive.")
    origin_policy = resolve_rollout_origin_policy(int(horizon_minutes), str(origin_policy))
    selection_target = resolve_rollout_selection_target(
        int(horizon_minutes),
        str(selection_target),
    )

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_root = output_root / "challenger_sweeps"
    sweep_dir = sweep_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "run_id": sweep_dir.name,
        "stage": "007_rollout_challenger_sweep",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "load_type": DATASET["load_type"],
        "artifact_namespace": DATASET["artifact_namespace"],
        "horizon_minutes": int(horizon_minutes),
        "origins_per_run": int(origins),
        "origin_policy": str(origin_policy),
        "selection_target": str(selection_target),
        "max_candidates": int(max_candidates),
        "origin_selection_scope": "shared_timestamp_intersection",
        "warnings": [],
        "failure_reason": "",
        "artifacts": {
            "candidate_plan": "candidate_plan.csv",
            "shared_origins": "shared_origins.csv",
            "candidate_results": "candidate_results.csv",
            "recommended_candidate": "recommended_candidate.json",
            "challenger_sweep_registry": "challenger_sweep_registry.csv",
            "challenger_summary_md": "challenger_summary.md",
        },
    }
    _write_sweep_manifest(sweep_dir, manifest)

    requested_origin_policies: list[str] = []
    candidate_plan = pd.DataFrame()
    shared_origins = pd.DataFrame()
    sweep_policy_results = pd.DataFrame()
    sweep_policy_metadata: list[dict[str, Any]] = []
    sweep_policy_by_origin = pd.DataFrame()
    plan_warnings: list[str] = []
    shared_origin_warnings: list[str] = []
    rollout_latest_warning = ""
    try:
        requested_origin_policies = _requested_origin_policies_for_sweep(str(origin_policy))
        manifest["origin_policies"] = requested_origin_policies
        _write_sweep_manifest(sweep_dir, manifest)

        candidate_plan_frames: list[pd.DataFrame] = []
        for requested_origin_policy in requested_origin_policies:
            plan = _build_challenger_plan(
                requested_horizon_minutes=int(horizon_minutes),
                requested_origin_policy=requested_origin_policy,
                selection_target=str(selection_target),
                output_root=output_root,
                max_candidates=int(max_candidates),
            )
            if plan.empty:
                continue
            plan = plan.copy()
            plan["requested_origin_policy"] = requested_origin_policy
            candidate_plan_frames.append(plan)
        candidate_plan = (
            pd.concat(candidate_plan_frames, ignore_index=True)
            if candidate_plan_frames
            else pd.DataFrame()
        )
        candidate_plan, plan_warnings = _partition_representable_candidates(
            candidate_plan,
            requested_horizon_minutes=int(horizon_minutes),
        )
        manifest["warnings"] = list(plan_warnings)
        manifest["candidate_count"] = int(len(candidate_plan))
        _write_sweep_manifest(sweep_dir, manifest)
        if candidate_plan.empty:
            raise RuntimeError(
                "No rollout challenger candidates were runnable at the requested horizon after "
                "representability filtering."
            )
        candidate_plan.to_csv(sweep_dir / "candidate_plan.csv", index=False, float_format="%.6f")

        shared_origins, shared_origin_warnings = _build_shared_origin_timestamps(
            candidate_plan=candidate_plan,
            requested_horizon_minutes=int(horizon_minutes),
            origins=int(origins),
        )
        manifest["warnings"] = [*plan_warnings, *shared_origin_warnings]
        manifest["shared_origin_count"] = int(len(shared_origins))
        _write_sweep_manifest(sweep_dir, manifest)
        if shared_origins.empty:
            raise RuntimeError(
                "No shared rollout origin timestamps were available across the selected challenger "
                "candidates."
            )
        shared_origins.to_csv(sweep_dir / "shared_origins.csv", index=False)
        shared_origin_map = {
            str(policy): [
                pd.Timestamp(value)
                for value in group["origin_timestamp"].astype("string").tolist()
            ]
            for policy, group in shared_origins.groupby("requested_origin_policy", dropna=False)
        }

        plan_rows = [row.to_dict() for _, row in candidate_plan.iterrows()]
        parallel_plan = resolve_parallel_plan("rollout_sweep", task_count=len(plan_rows))
        legacy_workers = int(max(1, min(int(MULTIRES_ROLLOUT_CHALLENGERS["parallel_workers"]), len(plan_rows))))
        max_workers = legacy_workers
        if supports_high_capacity_parallelism():
            max_workers = int(max(legacy_workers, min(int(parallel_plan.n_jobs), len(plan_rows))))
        manifest["legacy_parallel_workers"] = int(legacy_workers)
        manifest["parallel_workers"] = int(max_workers)
        manifest["parallel_plan"] = parallel_plan.as_dict()
        _write_sweep_manifest(sweep_dir, manifest)
        for row in plan_rows:
            logger.info(
                "Evaluating rollout challenger %d/%d: %s/%s/%s",
                int(row["candidate_rank"]),
                len(plan_rows),
                row["resolution"],
                row["feature_set"],
                row["model_label"],
            )
        if max_workers == 1:
            run_records = [
                _evaluate_challenger_plan_row(
                    output_root=output_root,
                    plan_row=row,
                    horizon_minutes=int(horizon_minutes),
                    origins=int(origins),
                    selection_target=str(selection_target),
                    shared_origin_map=shared_origin_map,
                )
                for row in plan_rows
            ]
        else:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rollout-sweep") as executor:
                run_records = list(
                    executor.map(
                        lambda row: _evaluate_challenger_plan_row(
                            output_root=output_root,
                            plan_row=row,
                            horizon_minutes=int(horizon_minutes),
                            origins=int(origins),
                            selection_target=str(selection_target),
                            shared_origin_map=shared_origin_map,
                        ),
                        plan_rows,
                    )
                )
        run_results = [
            cast(dict[str, Any], record["summary"])
            for record in run_records
        ]

        results = pd.DataFrame(run_results)
        sweep_policy_results, sweep_policy_metadata, sweep_policy_by_origin = (
            _build_cross_candidate_phase_bucket_policies(
                run_records=run_records,
                candidate_results=results,
                sweep_dir=sweep_dir,
                requested_horizon_minutes=int(horizon_minutes),
                requested_origin_policy=str(origin_policy),
            )
        )
        if not sweep_policy_results.empty:
            results = pd.concat([results, sweep_policy_results], ignore_index=True)
        recommended = _select_recommended_candidate(
            results,
            selection_target=str(selection_target),
            requested_origin_policy=str(origin_policy),
            recommendation_origin_scope=str(MULTIRES_ROLLOUT_CHALLENGERS["recommendation_origin_scope"]),
        )
        results = results.sort_values(
            ["candidate_rank", "selection_metric_value", "secondary_metric_value"],
            ascending=[True, True, True],
            kind="stable",
        ).reset_index(drop=True)
        results.to_csv(sweep_dir / "candidate_results.csv", index=False, float_format="%.6f")
        if sweep_policy_metadata:
            (sweep_dir / "portfolio_policy_candidates.json").write_text(
                json.dumps(sweep_policy_metadata, indent=2),
                encoding="utf-8",
            )
        if not sweep_policy_by_origin.empty:
            sweep_policy_by_origin.to_csv(
                sweep_dir / "portfolio_policy_by_origin.csv",
                index=False,
                float_format="%.6f",
            )
        _write_sweep_summary_md(
            results=results,
            recommended=recommended,
            output_dir=sweep_dir,
            selection_target=str(selection_target),
        )

        recommended_payload = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "load_type": DATASET["load_type"],
            "artifact_namespace": DATASET["artifact_namespace"],
            "selection_target": str(selection_target),
            "requested_horizon_minutes": int(horizon_minutes),
            "requested_origin_policies": requested_origin_policies,
            "recommended_origin_policy": str(recommended["requested_origin_policy"]),
            "recommended_candidate_label": str(recommended["candidate_label"]),
            "recommended_resolution": str(recommended["resolution"]),
            "recommended_feature_set": str(recommended["feature_set"]),
            "recommended_model_label": str(recommended["model_label"]),
            "recommended_target_mode": str(recommended.get("learned_target_mode", "")),
            "recommended_source_type": str(recommended.get("source_type", "")),
            "recommended_run_id": str(recommended["run_id"]),
            "recommended_run_path": str(recommended["run_path"]),
            "recommended_metric_value": float(recommended["selection_metric_value"]),
            "recommended_metric_pct": float(recommended["selection_metric_pct"]),
            "reason": (
                "Best challenger under the requested rollout objective after ranking by "
                "origin-policy match, baseline wins, coverage, and measured rollout error."
            ),
        }
        (sweep_dir / "recommended_candidate.json").write_text(
            json.dumps(recommended_payload, indent=2),
            encoding="utf-8",
        )

        sweep_registry = _build_challenger_sweep_registry_snapshot(output_root)
        sweep_registry.to_csv(sweep_dir / "challenger_sweep_registry.csv", index=False)
        sweep_registry.to_csv(output_root / "challenger_sweep_registry.csv", index=False)

        if refresh_rollout_registry:
            root_registry = _build_rollout_registry_snapshot(output_root)
            root_registry.to_csv(output_root / "rollout_registry.csv", index=False)
        if refresh_rollout_latest:
            best_run_dir = output_root / str(recommended["run_id"])
            if best_run_dir.exists():
                update_latest_alias(best_run_dir, output_root / "latest", enabled=bool(MULTIRES_CONFIG["write_latest"]))
            else:
                rollout_latest_warning = (
                    "skipped_rollout_latest_alias_for_sweep_derived_candidate:"
                    f"{recommended['candidate_label']}"
                )
                logger.info(
                    "Skipping rollout latest alias refresh because recommended candidate %s is sweep-derived "
                    "and does not correspond to a standalone rollout run directory.",
                    recommended["candidate_label"],
                )

        manifest.update(
            {
                "status": "success",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "origin_policies": requested_origin_policies,
                "shared_origin_count": int(len(shared_origins)),
                "recommendation_origin_scope": str(MULTIRES_ROLLOUT_CHALLENGERS["recommendation_origin_scope"]),
                "candidate_count": int(len(candidate_plan)),
                "portfolio_candidate_count": int(len(sweep_policy_results)),
                "warnings": [
                    *plan_warnings,
                    *shared_origin_warnings,
                    *([rollout_latest_warning] if rollout_latest_warning else []),
                ],
                "recommended_run_id": str(recommended["run_id"]),
                "recommended_candidate_label": str(recommended["candidate_label"]),
                "recommended_metric_value": float(recommended["selection_metric_value"]),
                "recommended_metric_pct": float(recommended["selection_metric_pct"]),
                "baseline_labels": _enabled_rollout_baselines(),
                "config_hash": stable_config_hash(
                    {
                        "horizon_minutes": int(horizon_minutes),
                        "origins": int(origins),
                        "origin_policy": str(origin_policy),
                        "selection_target": str(selection_target),
                        "max_candidates": int(max_candidates),
                        "challenger_config": dict(MULTIRES_ROLLOUT_CHALLENGERS),
                    }
                ),
            }
        )
        if sweep_policy_metadata:
            manifest["artifacts"]["portfolio_policy_candidates"] = "portfolio_policy_candidates.json"
        if not sweep_policy_by_origin.empty:
            manifest["artifacts"]["portfolio_policy_by_origin"] = "portfolio_policy_by_origin.csv"
        _write_sweep_manifest(sweep_dir, manifest)

        if refresh_sweep_latest:
            latest_dir = sweep_root / "latest"
            update_latest_alias(sweep_dir, latest_dir, enabled=bool(MULTIRES_CONFIG["write_latest"]))
            if bool(MULTIRES_CONFIG["write_latest"]):
                for artifact_name in (
                    "candidate_plan.csv",
                    "shared_origins.csv",
                    "candidate_results.csv",
                    "challenger_sweep_registry.csv",
                    "challenger_summary.md",
                    "recommended_candidate.json",
                    "run_manifest.json",
                ):
                    source = sweep_dir / artifact_name
                    if source.exists():
                        shutil.copy2(source, latest_dir / artifact_name)
                for artifact_name in ("portfolio_policy_candidates.json", "portfolio_policy_by_origin.csv"):
                    source = sweep_dir / artifact_name
                    if source.exists():
                        shutil.copy2(source, latest_dir / artifact_name)

        logger.info("Rollout challenger sweep artifacts written to %s", sweep_dir)
        return {
            "sweep_dir": sweep_dir,
            "candidate_plan": candidate_plan,
            "candidate_results": results,
            "recommended": recommended,
            "recommended_payload": recommended_payload,
            "manifest": manifest,
        }
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "origin_policies": requested_origin_policies,
                "candidate_count": int(len(candidate_plan)),
                "shared_origin_count": int(len(shared_origins)),
                "portfolio_candidate_count": int(len(sweep_policy_results)),
                "warnings": [*plan_warnings, *shared_origin_warnings],
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_sweep_manifest(sweep_dir, manifest)
        raise


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the Stage-7 challenger-sweep runner."""
    parser = argparse.ArgumentParser(description="Run a scheduled Stage-7 rollout challenger sweep.")
    parser.add_argument(
        "--horizon-minutes",
        type=int,
        default=MULTIRES_ROLLOUT["horizon_minutes"],
    )
    parser.add_argument(
        "--origins",
        type=int,
        default=MULTIRES_ROLLOUT["origins_per_run"],
    )
    parser.add_argument(
        "--origin-policy",
        choices=["uniform", "midnight", "billing_aligned", "phase_balanced", "auto"],
        default=MULTIRES_ROLLOUT["origin_policy"],
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
        default=MULTIRES_ROLLOUT["selection_target"],
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=MULTIRES_ROLLOUT_CHALLENGERS["max_candidates"],
    )
    parser.add_argument(
        "--output-dir",
        default=str(scoped_output_path(PATHS["outputs_rollout_dir"])),
    )
    return parser.parse_args()


def main() -> int:
    """Execute one challenger sweep and persist its registry-backed artifacts."""
    _configure_logging()
    validate_config()
    args = parse_args()
    if not MULTIRES_ROLLOUT_CHALLENGERS["enabled"]:
        logger.error("multires.rollout_challengers.enabled=false; sweep execution is disabled.")
        return 1
    if int(args.max_candidates) <= 0:
        raise ValueError("--max-candidates must be positive.")
    try:
        run_rollout_challenger_sweep(
            output_root=Path(args.output_dir).resolve(),
            horizon_minutes=int(args.horizon_minutes),
            origins=int(args.origins),
            origin_policy=str(args.origin_policy),
            selection_target=str(args.selection_target),
            max_candidates=int(args.max_candidates),
        )
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
