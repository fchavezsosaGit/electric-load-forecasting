"""Unit tests for model performance workflow helpers."""

from __future__ import annotations

import importlib.util
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _find_project_root(start: Path) -> Path:
    """Find the repository root so the Stage-5 module can be loaded in isolation."""
    for candidate in [start, *start.parents]:
        if (candidate / "run_pipeline.py").exists() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Unable to locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_performance_module():
    """Load the Stage-5 entry module directly from disk for helper-level tests."""
    path = SCRIPTS_DIR / "004_model_performance.py"
    spec = importlib.util.spec_from_file_location("test_step5_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_walkforward_folds_default_shape():
    """Build the expected expanding-train rolling-validation fold layout."""
    module = _load_performance_module()
    folds = module.build_walkforward_folds(
        holdout_start_day=29,
        n_folds=5,
        val_window_days=2,
        train_start_day=1,
    )

    assert len(folds) == 5
    assert folds[0] == {
        "fold": 1,
        "train_start_day": 1,
        "train_end_day": 18,
        "val_start_day": 19,
        "val_end_day": 20,
    }
    assert folds[-1] == {
        "fold": 5,
        "train_start_day": 1,
        "train_end_day": 26,
        "val_start_day": 27,
        "val_end_day": 28,
    }


def test_build_selection_scoreboard_sorts_by_ratio_then_std():
    """Rank the Stage-5 scoreboard by MAE ratio before fold-to-fold variability."""
    module = _load_performance_module()
    df = pd.DataFrame(
        [
            {"fold": 1, "resolution": "1min", "feature_set": "curated", "model_label": "a", "target_mode": "raw", "mae_ratio": 0.95, "mae": 100.0, "coverage": 1.0},
            {"fold": 2, "resolution": "1min", "feature_set": "curated", "model_label": "a", "target_mode": "raw", "mae_ratio": 0.99, "mae": 110.0, "coverage": 1.0},
            {"fold": 1, "resolution": "1min", "feature_set": "curated", "model_label": "b", "target_mode": "raw", "mae_ratio": 0.92, "mae": 99.0, "coverage": 1.0},
            {"fold": 2, "resolution": "1min", "feature_set": "curated", "model_label": "b", "target_mode": "raw", "mae_ratio": 0.97, "mae": 111.0, "coverage": 1.0},
        ]
    )
    scored = module.build_selection_scoreboard(df)
    assert list(scored["model_label"]) == ["b", "a"]
    assert list(scored.columns) == [
        "resolution",
        "feature_set",
        "model_label",
        "target_mode",
        "fold_mean_mae_ratio",
        "fold_std_mae_ratio",
        "fold_n",
        "raw_validate_mae",
        "raw_validate_mae_pct",
        "raw_validate_rmse_pct",
        "mean_coverage",
    ]


def test_build_residual_ablation_has_delta_columns():
    """Expose residual-vs-raw deltas for the same candidate in the ablation table."""
    module = _load_performance_module()
    scoreboard = pd.DataFrame(
        [
            {"resolution": "1min", "feature_set": "curated", "model_label": "ridge-medium", "target_mode": "raw", "fold_mean_mae_ratio": 0.98, "raw_validate_mae": 120.0},
            {"resolution": "1min", "feature_set": "curated", "model_label": "ridge-medium", "target_mode": "residual", "fold_mean_mae_ratio": 0.95, "raw_validate_mae": 115.0},
        ]
    )
    ablation = module.build_residual_ablation(scoreboard)
    assert len(ablation) == 1
    row = ablation.iloc[0]
    assert row["delta_fold_mean_mae_ratio"] == row["residual_fold_mean_mae_ratio"] - row["raw_fold_mean_mae_ratio"]
    assert row["delta_mean_mae"] == row["residual_mean_mae"] - row["raw_mean_mae"]


def test_select_promotion_candidate_prefers_coverage_guarded_row():
    """Prefer the coverage-safe candidate over a lower-error but low-coverage row."""
    module = _load_performance_module()
    scoreboard = pd.DataFrame(
        [
            {
                "resolution": "1min",
                "feature_set": "full",
                "model_label": "ridge-strong",
                "target_mode": "raw",
                "fold_mean_mae_ratio": 0.80,
                "fold_std_mae_ratio": 0.01,
                "fold_n": 5,
                "raw_validate_mae": 430.0,
                "mean_coverage": 0.37,
            },
            {
                "resolution": "1min",
                "feature_set": "curated_ramp",
                "model_label": "hgb-balanced",
                "target_mode": "residual+blend",
                "fold_mean_mae_ratio": 0.94,
                "fold_std_mae_ratio": 0.02,
                "fold_n": 5,
                "raw_validate_mae": 520.0,
                "mean_coverage": 0.99,
            },
        ]
    )

    promoted = module._select_promotion_candidate(scoreboard)

    assert promoted is not None
    assert promoted["feature_set"] == "curated_ramp"
    assert promoted["target_mode"] == "residual+blend"
    assert promoted["coverage_gate_passed"] is True


def test_derive_operating_regime_prefers_transition_then_activity():
    """Collapse the existing causal regime flags into one deployment-oriented regime label."""
    module = _load_performance_module()
    frame = pd.DataFrame(
        {
            "day_class": ["workday", "workday", None, "holiday"],
            "profile_active_flag": [1, 0, 0, 1],
            "workday_transition": [1, 1, 0, 0],
        }
    )

    regime = module._derive_operating_regime(frame)

    assert regime is not None
    assert list(regime.astype(str)) == [
        "transition_active",
        "transition_only",
        "inactive",
        "active_profile",
    ]


def test_build_stage5_operating_policy_can_emit_learned_regime_override():
    """Keep the global default honest while still allowing learned minute overrides where evidence exists."""
    module = _load_performance_module()
    deployment = {
        "recommended_candidate_label": "persistence",
        "recommended_candidate_type": "baseline",
        "learned_beats_best_baseline": False,
    }
    regime_evaluation = pd.DataFrame(
        [
            {
                "segment_column": "operating_regime",
                "segment_value": "active_profile",
                "candidate_label": "curated_ramp/hgb-balanced/residual+blend",
                "candidate_mae": 90.0,
                "persistence_mae": 100.0,
                "best_baseline_label": "persistence",
                "best_baseline_mae": 100.0,
                "candidate_mae_ratio_to_persistence": 0.90,
                "candidate_mae_ratio_to_best_baseline": 0.90,
                "rows": 12,
            },
            {
                "segment_column": "operating_regime",
                "segment_value": "inactive",
                "candidate_label": "curated_ramp/hgb-balanced/residual+blend",
                "candidate_mae": 110.0,
                "persistence_mae": 100.0,
                "best_baseline_label": "persistence",
                "best_baseline_mae": 100.0,
                "candidate_mae_ratio_to_persistence": 1.10,
                "candidate_mae_ratio_to_best_baseline": 1.10,
                "rows": 40,
            },
        ]
    )

    policy = module._build_stage5_operating_policy(
        deployment_recommendation=deployment,
        candidate_label="curated_ramp/hgb-balanced/residual+blend",
        best_baseline_label="persistence",
        operating_regime_evaluation=regime_evaluation,
    )

    assert policy["standalone_operating_role"] == "baseline_anchor"
    assert policy["stage10_operating_role"] == "corrective_overlay_specialist"
    assert len(policy["regime_overrides"]) == 1
    assert policy["regime_overrides"][0]["operating_regime"] == "active_profile"
    assert policy["regime_overrides"][0]["recommended_candidate_label"] == "curated_ramp/hgb-balanced/residual+blend"
    assert policy["regime_evidence"][1]["recommended_candidate_label"] == "persistence"


def test_build_holdout_coverage_summary_flags_narrow_regime_support():
    """Persist a clear warning when the promoted Stage-5 holdout only covers one regime slice."""
    module = _load_performance_module()
    holdout_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-12-26 00:00:00", periods=4, freq="6h"),
            "day_class": ["none"] * 4,
            "profile_active_flag": [0.0] * 4,
            "workday_transition": [0.0] * 4,
            "operating_regime": ["none_inactive"] * 4,
        }
    )

    coverage_segments, summary = module._build_holdout_coverage_summary(
        holdout_frame=holdout_frame,
        segment_columns=["day_class", "profile_active_flag", "workday_transition", "operating_regime"],
    )

    assert not coverage_segments.empty
    assert bool(summary["narrow_regime_support"]) is True
    assert summary["single_value_segment_columns"] == [
        "day_class",
        "profile_active_flag",
        "workday_transition",
        "operating_regime",
    ]
    assert summary["dominant_operating_regime"] == "none_inactive"


def test_prediction_surface_segment_evaluation_adds_priority_bands():
    """Supplemental Stage-5 surfaces should emit regime, load-band, ramp-band, and source slices."""
    module = _load_performance_module()
    prediction_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-12-20 00:00:00", periods=4, freq="1min"),
            "y_true": [100.0, 240.0, 160.0, 280.0],
            "candidate_pred": [102.0, 220.0, 158.0, 250.0],
            "persistence_pred": [98.0, 260.0, 170.0, 300.0],
            "previous_day_pred": [110.0, 250.0, 175.0, 305.0],
            "day_class": ["workday", "workday", "none", "none"],
            "profile_active_flag": [0.0, 1.0, 0.0, 1.0],
            "workday_transition": [0.0, 1.0, 0.0, 0.0],
            "evaluation_surface": ["validate_walkforward", "validate_walkforward", "test_holdout", "test_holdout"],
        }
    )
    prediction_columns = {
        "candidate/model/raw": "candidate_pred",
        "persistence": "persistence_pred",
        "previous_day": "previous_day_pred",
    }

    segment_evaluation, operating_regime_evaluation, coverage_segments, coverage_summary = (
        module._prediction_surface_segment_evaluation(
            prediction_frame=prediction_frame,
            prediction_columns=prediction_columns,
            candidate_label="candidate/model/raw",
            best_baseline_label="previous_day",
        )
    )

    assert {"operating_regime", "actual_load_band", "actual_ramp_band", "evaluation_surface"}.issubset(
        set(segment_evaluation["segment_column"])
    )
    assert not operating_regime_evaluation.empty
    assert not coverage_segments.empty
    assert "actual_load_band" in coverage_summary["segment_columns"]


def test_build_supplemental_surface_artifacts_stitches_validate_and_holdout_rows(monkeypatch):
    """Supplemental Stage-5 evidence should combine walk-forward validate rows with the canonical holdout."""
    module = _load_performance_module()
    validate_start, _validate_end = module.SPLIT_DAY_RANGES["validate"]
    holdout_start, _holdout_end = module.SPLIT_DAY_RANGES["test"]
    gold = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-12-20 00:00:00", periods=4, freq="1min"),
            "day_idx": [validate_start, validate_start, holdout_start, holdout_start],
            "day_class": ["workday", "workday", "none", "none"],
            "profile_active_flag": [0.0, 1.0, 0.0, 0.0],
            "workday_transition": [0.0, 1.0, 0.0, 0.0],
        }
    )
    folds = [
        {
            "fold": 1,
            "train_start_day": 1,
            "train_end_day": validate_start - 1,
            "val_start_day": validate_start,
            "val_end_day": validate_start,
        }
    ]
    holdout_predictions = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-12-29 00:00:00", periods=2, freq="1min"),
            "y_true": [150.0, 170.0],
            "minimal_ridge_medium_raw_pred": [148.0, 169.0],
            "persistence_pred": [152.0, 175.0],
            "previous_day_pred": [155.0, 176.0],
            "day_class": ["none", "none"],
            "profile_active_flag": [0.0, 0.0],
            "workday_transition": [0.0, 0.0],
        }
    )
    holdout_prediction_columns = {
        "minimal/ridge-medium/raw": "minimal_ridge_medium_raw_pred",
        "persistence": "persistence_pred",
        "previous_day": "previous_day_pred",
    }
    validate_eval_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-12-22 00:00:00", periods=2, freq="1min"),
            "day_class": ["workday", "workday"],
            "profile_active_flag": [0.0, 1.0],
            "workday_transition": [0.0, 1.0],
        }
    )
    validate_payload = {
        "fold_meta": folds[0],
        "train_df": pd.DataFrame({"timestamp": pd.date_range("2025-12-01", periods=2, freq="1min")}),
        "eval_df": validate_eval_df,
        "aligned": pd.DataFrame(
            {
                "y_true": [100.0, 220.0],
                "y_pred": [102.0, 210.0],
                "y_persist": [98.0, 240.0],
            },
            index=[0, 1],
        ),
        "n_eval_total": 2,
    }

    monkeypatch.setattr(module, "_build_stage5_eval_payloads", lambda **_kwargs: [validate_payload])
    monkeypatch.setattr(
        module,
        "_stage5_surface_candidate_predictions",
        lambda **_kwargs: (
            pd.Series([102.0, 210.0], index=[0, 1], dtype=float),
            {"mae": 6.0, "rmse": 7.0, "mae_pct": 3.0, "rmse_pct": 3.5, "coverage": 1.0, "n_eval": 2},
            None,
        ),
    )
    monkeypatch.setattr(
        module,
        "_build_holdout_baseline_predictions",
        lambda **_kwargs: {
            "persistence": pd.Series([98.0, 240.0], index=[0, 1], dtype=float),
            "previous_day": pd.Series([110.0, 235.0], index=[0, 1], dtype=float),
        },
    )

    supplemental = module._build_supplemental_surface_artifacts(
        gold=gold,
        folds=folds,
        feature_sets={"minimal": ["day_class"]},
        model_catalog={"ridge-medium": module.build_model_catalog()["ridge-medium"]},
        resolution="1min",
        promoted_candidate={"feature_set": "minimal", "model_label": "ridge-medium", "target_mode": "raw"},
        steps_per_day=2,
        blend_config=None,
        holdout_predictions=holdout_predictions,
        holdout_prediction_columns=holdout_prediction_columns,
    )

    assert bool(supplemental["advisory"]["canonical_holdout_preserved"]) is True
    assert supplemental["advisory"]["evaluation_surface_counts"]["validate_walkforward"] == 2
    assert supplemental["advisory"]["evaluation_surface_counts"]["test_holdout"] == 2
    assert not supplemental["summary"].empty
    assert not supplemental["source_evaluation"].empty


def test_evaluate_promoted_holdout_candidate_outputs_holdout_comparison():
    """Return the holdout comparison, deployment decision, and segment diagnostics."""
    module = _load_performance_module()
    timestamps = pd.date_range("2025-12-01 00:00:00", periods=31 * 4, freq="6h")
    avg_load = 100.0 + np.linspace(0.0, 50.0, len(timestamps)) + np.sin(np.arange(len(timestamps)) / 3.0)
    gold = pd.DataFrame(
        {
            "timestamp": timestamps,
            "day_idx": np.repeat(np.arange(1, 32), 4),
            "avg_load": avg_load,
            "lag_1": pd.Series(avg_load).shift(1),
            "workday": 2,
            "hour": timestamps.hour,
            "day_of_week": ((timestamps.dayofweek + 1) % 7).astype(int),
            "season": 1,
            "time_of_day": 0,
        }
    )
    feature_sets = {"minimal": ["workday", "hour", "lag_1"]}
    promoted_candidate = {
        "feature_set": "minimal",
        "model_label": "ridge-medium",
        "target_mode": "raw",
    }

    holdout_summary, holdout_blend_decisions, deployment, segment_evaluation = module._evaluate_promoted_holdout_candidate(
        gold=gold,
        feature_sets=feature_sets,
        model_catalog={"ridge-medium": module.build_model_catalog()["ridge-medium"]},
        resolution="1min",
        promoted_candidate=promoted_candidate,
        steps_per_day=4,
        blend_config=None,
    )

    assert holdout_blend_decisions is None
    assert segment_evaluation.empty
    assert {"minimal/ridge-medium/raw", "persistence"}.issubset(set(holdout_summary["candidate_label"]))
    assert "holt_damped" in set(holdout_summary["candidate_label"])
    assert {"recommended_candidate_label", "recommended_candidate_type", "decision_reason"}.issubset(
        set(deployment)
    )
    assert "best_baseline_label" in deployment


def test_bootstrap_comparison_rows_emit_ci_and_significance_columns():
    """Summarize one paired holdout comparison with block-bootstrap intervals."""
    module = _load_performance_module()
    rows = module._bootstrap_comparison_rows(
        y_true=pd.Series(np.linspace(100.0, 112.0, 16)),
        candidate_pred=pd.Series(np.linspace(99.0, 111.0, 16)),
        baseline_pred=pd.Series(np.linspace(98.0, 110.0, 16)),
        candidate_label="learned",
        baseline_label="persistence",
        comparison_type="candidate_vs_persistence",
        resolution="1min",
        seed=42,
    )

    inference = pd.DataFrame(rows)
    assert set(inference["metric_name"]) == {"mae", "rmse"}
    assert (inference["bootstrap_block_length_steps"] >= 1).all()
    assert {"candidate_metric_ci_low", "candidate_metric_ci_high", "one_sided_p_candidate_lt_baseline"}.issubset(
        set(inference.columns)
    )


def test_compute_holdout_feature_importance_summarizes_top_features():
    """Rank holdout permutation importance and report concentration in the top features."""
    module = _load_performance_module()
    x_eval = pd.DataFrame(
        {
            "lag_1": np.linspace(100.0, 120.0, 40),
            "phase_progress_15m": np.tile(np.linspace(0.0, 1.0, 10), 4),
            "profile_activity_ratio": np.linspace(0.0, 1.0, 40),
        }
    )
    y_true = 0.9 * x_eval["lag_1"] + 8.0 * x_eval["profile_activity_ratio"]
    model = module.build_model_catalog()["ridge-medium"].factory()
    model.fit(x_eval, y_true)

    importance, summary = module._compute_holdout_feature_importance(
        model=model,
        x_eval=x_eval,
        y_true=y_true,
        target_mode="raw",
    )

    assert not importance.empty
    assert summary is not None
    assert summary["top_feature"] in set(importance["feature"])
    assert summary["top_5_cumulative_share"] >= 0.0


def test_build_feature_sets_includes_curated_ramp_without_duplicates():
    """Add the curated-ramp helper set without duplicating shared feature columns."""
    module = _load_performance_module()
    feature_sets = module._build_feature_sets(include_curated_ramp=True)
    assert module.RAMP_FEATURE_SET_NAME in feature_sets
    curated_ramp = feature_sets[module.RAMP_FEATURE_SET_NAME]
    assert len(curated_ramp) == len(set(curated_ramp))
    for required in module.RAMP_ADDITIONAL_FEATURES:
        assert required in curated_ramp


def test_build_feature_sets_includes_full_stable_without_long_rolling_windows():
    """Build the full-stable feature set without the excluded long-window columns."""
    module = _load_performance_module()
    feature_sets = module._build_feature_sets(include_curated_ramp=False)
    assert module.FULL_STABLE_FEATURE_SET_NAME in feature_sets
    full_stable = feature_sets[module.FULL_STABLE_FEATURE_SET_NAME]
    assert full_stable
    for excluded in module.FULL_STABLE_EXCLUDED_COLUMNS:
        assert excluded not in full_stable
    assert "lag_1440" in full_stable
    assert len(full_stable) < len(feature_sets["full"])


def test_build_feature_sets_includes_legacy_full_stable_variant():
    """Expose the older sparse full-stable recipe as an explicit feature family."""
    module = _load_performance_module()
    feature_sets = module._build_feature_sets(include_curated_ramp=False)
    assert module.FULL_STABLE_LEGACY_FEATURE_SET_NAME in feature_sets
    legacy = feature_sets[module.FULL_STABLE_LEGACY_FEATURE_SET_NAME]
    assert legacy == list(module.FULL_STABLE_LEGACY_COLUMNS)
    assert "rolling_mean_240" not in legacy
    assert "profile_activity_ratio" not in legacy
    assert "lag_1440" in legacy


def test_augment_with_curated_ramp_features_is_causal():
    """Derive curated-ramp features only from historical values and calendar context."""
    module = _load_performance_module()
    gold = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-12-01", periods=6, freq="1min"),
            "avg_load": [100.0, 120.0, 130.0, 110.0, 125.0, 150.0],
            "lag_1": [pd.NA, 100.0, 120.0, 130.0, 110.0, 125.0],
            "lag_5": [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 100.0],
            "delta_5": [pd.NA, 1.0, -2.0, 3.0, -4.0, 5.0],
            "hour": [0, 0, 0, 0, 0, 0],
        }
    )
    out, threshold = module._augment_with_curated_ramp_features(gold, ramp_quantile=0.8)
    assert threshold > 0.0
    assert round(float(out.loc[3, "rolling_mean_3"]), 6) == round((100.0 + 120.0 + 130.0) / 3.0, 6)
    assert pd.isna(out.loc[0, "ramp_flag"])
    assert out.loc[5, "hour_x_delta_5"] == 0.0


def test_hgb_coordinate_specs_has_required_coverage():
    """Expose the bounded coordinate-search HGB surface expected by Stage-5."""
    module = _load_performance_module()
    specs = module._hgb_coordinate_specs()
    assert len(specs) >= 12
    assert "hgb-coordinate-leaf50" in specs
    assert "hgb-coordinate-l2100" in specs


def test_hgb_frontier_specs_are_available():
    """Expose the hand-picked frontier HGB variants used downstream."""
    module = _load_performance_module()
    specs = module._hgb_frontier_specs()
    assert "hgb-frontier-lr010-leaf100" in specs
    assert "hgb-frontier-lr010-depth5-leaf100-l2001" in specs


def test_apply_blend_policy_outputs_guardrail_metrics():
    """Return blend decisions and guardrail metrics for a blended holdout candidate."""
    module = _load_performance_module()
    aligned = pd.DataFrame(
        {
            "y_true": [10.0, 11.0, 12.0, 13.0],
            "y_pred": [10.2, 10.9, 12.1, 12.8],
            "y_persist": [9.0, 10.0, 11.0, 12.0],
        },
        index=[1, 2, 3, 4],
    )
    config = module.BlendConfig(window=2, sharpness=6.0, min_weight=0.1, max_weight=0.9)
    metrics, decisions = module._apply_blend_policy(aligned=aligned, blend_config=config, n_eval_total=4)
    assert metrics["n_eval"] == 4
    assert len(decisions) == 4
    assert decisions["blend_weight"].between(0.1, 0.9).all()
    assert decisions.iloc[0]["blend_weight"] == 0.5


def test_bucket_blend_policy_calibrates_minute_bucket_weights():
    """Select a fixed minute-bucket policy when different parts of the lock interval want different anchors."""
    module = _load_performance_module()
    aligned = pd.DataFrame(
        {
            "y_true": [14.0, 10.0, 6.0, 14.0, 10.0, 6.0],
            "y_pred": [14.0, 6.0, 6.0, 14.0, 6.0, 6.0],
            "y_persist": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        },
        index=[10, 11, 12, 13, 14, 15],
    )
    timestamps = pd.to_datetime(
        [
            "2026-03-01T00:00:00",
            "2026-03-01T00:05:00",
            "2026-03-01T00:10:00",
            "2026-03-01T00:15:00",
            "2026-03-01T00:20:00",
            "2026-03-01T00:25:00",
        ]
    )

    bucket_config = module._calibrate_bucket_blend_config(
        aligned=aligned,
        timestamps=timestamps,
        bucket_size_minutes=5,
        cycle_minutes=15,
        candidate_weights=[0.0, 1.0],
    )

    assert bucket_config is not None
    assert bucket_config.weight_map() == {0: 1.0, 5: 0.0, 10: 1.0}

    metrics, decisions = module._apply_bucket_blend_policy(
        aligned=aligned,
        timestamps=timestamps,
        bucket_config=bucket_config,
        n_eval_total=len(aligned),
    )

    assert metrics["mae"] == 0.0
    assert list(decisions["blend_bucket"]) == [0, 5, 10, 0, 5, 10]
    assert list(decisions["blend_weight"]) == [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def test_classical_benchmarks_normalize_gapped_indices_before_statsmodels_fit():
    """Avoid unsupported-index warnings by resetting gapped indices before fitting statsmodels baselines."""
    module = _load_performance_module()
    values = 120.0 + np.linspace(0.0, 40.0, 96) + np.sin(np.arange(96) / 5.0)
    gapped = pd.Series(values, index=np.arange(200, 392, 2))

    with warnings.catch_warnings(record=True) as holt_caught:
        warnings.simplefilter("always")
        holt = module._holt_damped_forecast(gapped, horizon_steps=12)
    with warnings.catch_warnings(record=True) as arima_caught:
        warnings.simplefilter("always")
        arima = module._arima_forecast(gapped, horizon_steps=12)

    warning_messages = [str(item.message).lower() for item in [*holt_caught, *arima_caught]]
    assert holt is not None
    assert arima is not None
    assert not any("unsupported index" in message for message in warning_messages)
    assert not any("no supported index" in message for message in warning_messages)


def test_evaluate_blend_candidate_considers_xgb_labeled_rows(monkeypatch):
    """Allow optional XGB candidates to participate in the guarded Stage-5 blend search."""
    module = _load_performance_module()

    def _fake_fit_and_align(*, eval_df, model_spec, **_kwargs):
        y_true = eval_df["avg_load"].astype(float)
        y_persist = eval_df["lag_1"].astype(float)
        if model_spec.model_label == "xgb-synthetic":
            y_pred = y_true - 1.0
        else:
            y_pred = y_persist + 35.0
        aligned = pd.DataFrame(
            {"y_true": y_true, "y_pred": y_pred, "y_persist": y_persist},
            index=eval_df.index,
        )
        return aligned, 0.0

    monkeypatch.setattr(module, "_fit_and_align", _fake_fit_and_align)

    gold = pd.DataFrame(
        {
            "day_idx": np.repeat([1, 2, 3, 4], 3),
            "avg_load": [100.0, 104.0, 108.0, 130.0, 134.0, 138.0, 170.0, 174.0, 178.0, 210.0, 214.0, 218.0],
            "lag_1": [95.0, 100.0, 104.0, 108.0, 130.0, 134.0, 138.0, 170.0, 174.0, 178.0, 210.0, 214.0],
        }
    )
    selection_scoreboard = pd.DataFrame(
        [
            {
                "resolution": "1min",
                "feature_set": "minimal",
                "model_label": "xgb-synthetic",
                "target_mode": "raw",
                "fold_mean_mae_ratio": 0.90,
                "fold_std_mae_ratio": 0.01,
                "raw_validate_mae": 10.0,
            },
            {
                "resolution": "1min",
                "feature_set": "minimal",
                "model_label": "hgb-synthetic",
                "target_mode": "raw",
                "fold_mean_mae_ratio": 0.91,
                "fold_std_mae_ratio": 0.02,
                "raw_validate_mae": 11.0,
            },
        ]
    )
    feature_sets = {"minimal": ["lag_1"]}
    model_catalog = {
        "xgb-synthetic": module.ModelSpec(
            model_label="xgb-synthetic",
            family="xgb",
            params={"device": "cuda"},
            factory=lambda: object(),
        ),
        "hgb-synthetic": module.ModelSpec(
            model_label="hgb-synthetic",
            family="hgb",
            params={"max_depth": 3},
            factory=lambda: object(),
        ),
    }

    blend_df, decisions_df, candidate_meta, selected_config, blend_finalists = module._evaluate_blend_candidate(
        gold=gold,
        folds=[{"fold": 1, "train_start_day": 1, "train_end_day": 2, "val_start_day": 3, "val_end_day": 4}],
        selection_scoreboard=selection_scoreboard,
        feature_sets=feature_sets,
        model_catalog=model_catalog,
        resolution="1min",
        base_blend_config=module.BlendConfig(window=2, sharpness=4.0, min_weight=0.1, max_weight=0.9),
        steps_per_day=3,
    )

    assert not blend_df.empty
    assert not decisions_df.empty
    assert selected_config is not None
    assert candidate_meta is not None
    assert set(blend_finalists["model_label"]) == {"xgb-synthetic", "hgb-synthetic"}
    assert candidate_meta["model_label"] == "xgb-synthetic"


def test_run_fold_metric_task_uses_selected_model_catalog_for_adaptive_specs(monkeypatch):
    """Keep full Stage-5 runs stable when adaptive HGB search injects one-off model labels."""
    module = _load_performance_module()
    custom_spec = module.ModelSpec(
        model_label="hgb-auto-custom",
        family="hgb",
        params={"max_depth": 7},
        factory=lambda: object(),
    )
    gold = pd.DataFrame(
        {
            "day_idx": [1, 2],
            "avg_load": [100.0, 110.0],
            "lag_1": [95.0, 100.0],
        }
    )
    fallback_called = {"value": False}

    def _fake_build_model_catalog(**_kwargs):
        fallback_called["value"] = True
        return {}

    monkeypatch.setattr(module, "build_model_catalog", _fake_build_model_catalog)
    monkeypatch.setattr(
        module,
        "_fit_and_evaluate",
        lambda **_kwargs: {
            "mae": 1.0,
            "rmse": 1.0,
            "mae_pct": 1.0,
            "rmse_pct": 1.0,
            "mae_ratio": 0.9,
            "coverage": 1.0,
            "train_mae": 1.0,
            "train_val_mae_ratio": 1.0,
            "n_eval": 1,
            "n_eval_total": 1,
        },
    )

    row = module._run_fold_metric_task(
        module.FoldMetricTask(
            fold={"fold": 1, "train_start_day": 1, "train_end_day": 1, "val_start_day": 2, "val_end_day": 2},
            feature_set="minimal",
            model_label="hgb-auto-custom",
            target_mode="raw",
        ),
        gold=gold,
        feature_sets={"minimal": ["lag_1"]},
        resolution="1min",
        steps_per_day=1,
        model_catalog={"hgb-auto-custom": custom_spec},
        include_hgb_coordinate_search=False,
    )

    assert row is not None
    assert row["model_label"] == "hgb-auto-custom"
    assert fallback_called["value"] is False


def test_sigmoid_is_numerically_stable_for_extreme_inputs():
    """Keep the blend sigmoid stable even for very large-magnitude scores."""
    module = _load_performance_module()
    assert 0.0 <= module._sigmoid(1000.0) <= 1.0
    assert 0.0 <= module._sigmoid(-1000.0) <= 1.0


def test_build_hgb_coordinate_summary_returns_recommended_candidate():
    """Recommend the strongest bounded HGB variant from fold evidence."""
    module = _load_performance_module()
    df = pd.DataFrame(
        [
            {"fold": 1, "feature_set": "full", "target_mode": "raw", "model_label": "hgb-aggressive", "mae_ratio": 0.90, "train_val_mae_ratio": 0.85},
            {"fold": 2, "feature_set": "full", "target_mode": "raw", "model_label": "hgb-aggressive", "mae_ratio": 0.88, "train_val_mae_ratio": 0.86},
            {"fold": 1, "feature_set": "full", "target_mode": "raw", "model_label": "hgb-coordinate-leaf100", "mae_ratio": 0.84, "train_val_mae_ratio": 0.90},
            {"fold": 2, "feature_set": "full", "target_mode": "raw", "model_label": "hgb-coordinate-leaf100", "mae_ratio": 0.83, "train_val_mae_ratio": 0.91},
        ]
    )
    summary, recommended = module.build_hgb_coordinate_summary(df)
    assert not summary.empty
    assert recommended is not None
    assert recommended["model_label"] == "hgb-coordinate-leaf100"


def test_stage5_visualizations_write_pngs(tmp_path):
    """Write non-empty Stage-5 figure artifacts for the selection, CI, importance, and tradeoff plots."""
    module = _load_performance_module()
    scoreboard = pd.DataFrame(
        [
            {
                "resolution": "1min",
                "feature_set": "full",
                "model_label": "hgb-frontier-lr010-leaf100",
                "target_mode": "raw",
                "fold_mean_mae_ratio": 0.80,
                "fold_std_mae_ratio": 0.02,
                "fold_n": 5,
                "raw_validate_mae": 520.0,
                "mean_coverage": 0.99,
            },
            {
                "resolution": "1min",
                "feature_set": "full",
                "model_label": "hgb-frontier-lr010-depth5-leaf100-l2001",
                "target_mode": "raw+blend",
                "fold_mean_mae_ratio": 0.79,
                "fold_std_mae_ratio": 0.03,
                "fold_n": 5,
                "raw_validate_mae": 515.0,
                "mean_coverage": 0.99,
            },
        ]
    )
    hgb_summary = pd.DataFrame(
        [
            {
                "model_label": "hgb-frontier-lr010-leaf100",
                "fold_mean_mae_ratio": 0.80,
                "fold_std_mae_ratio": 0.02,
                "fold_n": 5,
                "mean_train_val_mae_ratio": 0.95,
                "train_val_gap_to_one": 0.05,
                "delta_mean_mae_ratio": -0.01,
                "delta_std_mae_ratio": -0.01,
                "delta_train_val_gap": -0.01,
                "meets_p1b_acceptance": True,
            }
        ]
    )

    selection_path = tmp_path / "fig_selection_frontier.png"
    tradeoff_path = tmp_path / "fig_hgb_tradeoff.png"
    ci_path = tmp_path / "fig_holdout_benchmark_ci.png"
    importance_path = tmp_path / "fig_feature_importance.png"
    holdout_summary = pd.DataFrame(
        [
            {
                "candidate_label": "learned",
                "candidate_type": "promoted_learned",
                "mae": 150.0,
                "mae_pct": 7.0,
            },
            {
                "candidate_label": "persistence",
                "candidate_type": "baseline",
                "mae": 160.0,
                "mae_pct": 7.5,
            },
        ]
    )
    holdout_inference = pd.DataFrame(
        [
            {
                "metric_name": "mae",
                "candidate_label": "learned",
                "candidate_metric": 150.0,
                "candidate_metric_ci_low": 145.0,
                "candidate_metric_ci_high": 155.0,
                "baseline_label": "persistence",
                "baseline_metric": 160.0,
                "baseline_metric_ci_low": 156.0,
                "baseline_metric_ci_high": 164.0,
            }
        ]
    )
    feature_importance = pd.DataFrame(
        [
            {"feature": "lag_1", "importance_mean": 10.0, "importance_std": 1.0},
            {"feature": "profile_activity_ratio", "importance_mean": 4.0, "importance_std": 0.5},
        ]
    )
    module._write_selection_frontier_figure(scoreboard, selection_path)
    module._write_hgb_tradeoff_figure(hgb_summary, tradeoff_path)
    module._write_holdout_benchmark_ci_figure(holdout_summary, holdout_inference, ci_path)
    module._write_feature_importance_figure(
        feature_importance,
        candidate_label="learned",
        output_path=importance_path,
    )

    assert selection_path.exists()
    assert selection_path.stat().st_size > 0
    assert tradeoff_path.exists()
    assert tradeoff_path.stat().st_size > 0
    assert ci_path.exists()
    assert ci_path.stat().st_size > 0
    assert importance_path.exists()
    assert importance_path.stat().st_size > 0


def test_prepare_output_run_dir_creates_timestamped_subdirectory(tmp_path):
    """Create a timestamped run directory beneath the requested output root."""
    module = _load_performance_module()

    run_dir = module._prepare_output_run_dir(tmp_path)

    assert run_dir.parent == tmp_path
    assert run_dir.exists()
    assert run_dir.name.endswith("Z")


def test_write_preflight_manifest_updates_latest_alias(tmp_path):
    """Write the preflight manifest and refresh the Stage-5 latest alias."""
    module = _load_performance_module()
    output_root = tmp_path / "outputs"
    run_dir = module._prepare_output_run_dir(output_root)
    (run_dir / "feature_causality_audit.csv").write_text("status\npass\n", encoding="utf-8")
    (run_dir / "minute_integrity_audit.csv").write_text("status\npass\n", encoding="utf-8")
    (run_dir / "holdout_lock.json").write_text("{}", encoding="utf-8")
    (run_dir / "preflight_audit.md").write_text("# preflight\n", encoding="utf-8")
    model = module.build_model_catalog()["ridge-medium"]

    module._write_preflight_manifest(
        output_dir=run_dir,
        output_root=output_root,
        resolution="1min",
        selected_feature_sets=["curated"],
        selected_models=[model],
        preflight={"overall_status": "pass"},
    )

    manifest_path = run_dir / "run_manifest.json"
    latest_manifest = output_root / "latest" / "run_manifest.json"
    assert manifest_path.exists()
    assert latest_manifest.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    latest_text = latest_manifest.read_text(encoding="utf-8")
    assert manifest["stage"] == "005_performance"
    assert manifest["mode"] == "preflight"
    assert '"stage": "005_performance"' in latest_text
    assert '"mode": "preflight"' in latest_text


def test_build_stage5_holdout_registry_backfills_cross_run_holdout_winners(tmp_path):
    """Summarize historical Stage-5 holdout artifacts into a root-level registry."""
    module = _load_performance_module()
    run_dir = tmp_path / "20260312T000000000000Z"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "holdout_evaluation.csv").write_text(
        "\n".join(
            [
                "candidate_label,candidate_type,resolution,feature_set,model_label,target_mode,mae,rmse,mae_pct,rmse_pct,mae_ratio_to_persistence,coverage,n_eval,n_eval_total",
                "full_stable/hgb-frontier-lr010-l2001/raw+blend,promoted_learned,1min,full_stable,hgb-frontier-lr010-l2001,raw+blend,163.0,233.6,7.86,11.27,0.938,1.0,4320,4320",
                "persistence,baseline,1min,baseline,persistence,raw,173.7,270.9,8.38,13.07,1.0,1.0,4320,4320",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "deployment_recommendation.json").write_text(
        json.dumps(
            {
                "recommended_candidate_label": "full_stable/hgb-frontier-lr010-l2001/raw+blend",
                "recommended_candidate_type": "promoted_learned",
                "decision_reason": "Promoted Stage-5 candidate beat persistence on holdout MAE.",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "generated_at_utc": "2026-03-12T00:00:00+00:00",
                "mode": "full",
                "blend_policy": {
                    "enabled": True,
                    "window": 240,
                    "sharpness": 9.0,
                    "min_weight": 0.0,
                    "max_weight": 1.0,
                    "candidate": {
                        "feature_set": "full_stable",
                        "model_label": "hgb-frontier-lr010-l2001",
                        "target_mode": "raw",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    registry = module.build_stage5_holdout_registry(tmp_path)

    assert len(registry) == 1
    row = registry.iloc[0]
    assert str(row["learned_candidate_label"]) == "full_stable/hgb-frontier-lr010-l2001/raw+blend"
    assert bool(row["learned_beats_persistence"]) is True
    assert float(row["blend_window"]) == 240.0
    assert str(row["recommended_candidate_type"]) == "promoted_learned"
