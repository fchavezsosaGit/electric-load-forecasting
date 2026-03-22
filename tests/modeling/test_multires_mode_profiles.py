"""Unit tests for config-driven multiresolution execution profiles."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_multires_module():
    """Load the Stage-6 entry module from disk for configuration tests."""
    path = Path("scripts/005_multires_compare.py").resolve()
    spec = importlib.util.spec_from_file_location("test_multires_compare_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_mode_profile_uses_configured_profile_scopes():
    """Ensure each named Stage-6 mode resolves to the expected configured scope."""
    module = _load_multires_module()

    smoke = module._resolve_mode_profile("smoke")
    candidate = module._resolve_mode_profile("candidate")
    full = module._resolve_mode_profile("full")
    focus_60m = module._resolve_mode_profile("focus_60m")

    assert smoke["resolutions"] == ["30s", "1min"]
    assert smoke["horizons_minutes"] == [15, 60]
    assert smoke["feature_sets"] == ["minimal", "curated"]
    assert candidate["resolutions"] == ["10s", "30s", "1min", "5min", "10min", "15min"]
    assert candidate["feature_sets"] == ["minimal", "curated", "full_stable"]
    assert "hgb-frontier-lr010-l2001" in candidate["model_labels"]
    assert "hgb-frontier-lr010-leaf100" in candidate["model_labels"]
    assert full["resolutions"] == ["1s", "5s", "10s", "30s", "1min", "5min", "10min", "15min"]
    assert all(resolution in full["resolutions"] for resolution in ["1s", "5s", "10s", "30s"])
    assert full["feature_sets"] == ["minimal", "curated", "full_stable"]
    assert "hgb-frontier-lr010-depth5-leaf100-l2001" in full["model_labels"]
    assert focus_60m["resolutions"] == ["30s", "1min", "5min", "10min"]
    assert focus_60m["feature_sets"] == ["minimal", "curated", "full_stable"]
    assert "hgb-balanced" in focus_60m["model_labels"]
    assert "hgb-frontier-lr010-l2001" in focus_60m["model_labels"]
    assert "hgb-frontier-lr010-leaf100" in focus_60m["model_labels"]
    assert "xgb-balanced" in focus_60m["model_labels"]
