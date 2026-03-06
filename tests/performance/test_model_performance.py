"""Unit tests for model performance workflow helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "run_pipeline.py").exists() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Unable to locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_performance_module():
    path = SCRIPTS_DIR / "004_model_performance.py"
    spec = importlib.util.spec_from_file_location("test_step5_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_walkforward_folds_default_shape():
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
        "mean_coverage",
    ]


def test_build_residual_ablation_has_delta_columns():
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


def test_build_feature_sets_includes_curated_ramp_without_duplicates():
    module = _load_performance_module()
    feature_sets = module._build_feature_sets(include_curated_ramp=True)
    assert module.RAMP_FEATURE_SET_NAME in feature_sets
    curated_ramp = feature_sets[module.RAMP_FEATURE_SET_NAME]
    assert len(curated_ramp) == len(set(curated_ramp))
    for required in module.RAMP_ADDITIONAL_FEATURES:
        assert required in curated_ramp


def test_augment_with_curated_ramp_features_is_causal():
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
    module = _load_performance_module()
    specs = module._hgb_coordinate_specs()
    assert len(specs) >= 12
    assert "hgb-coordinate-leaf50" in specs
    assert "hgb-coordinate-l2100" in specs


def test_apply_blend_policy_outputs_guardrail_metrics():
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


def test_sigmoid_is_numerically_stable_for_extreme_inputs():
    module = _load_performance_module()
    assert 0.0 <= module._sigmoid(1000.0) <= 1.0
    assert 0.0 <= module._sigmoid(-1000.0) <= 1.0


def test_build_hgb_coordinate_summary_returns_recommended_candidate():
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
