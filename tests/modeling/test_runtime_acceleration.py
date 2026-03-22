"""Tests for optional acceleration and runtime hardware helpers."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from scripts.modeling import common as common_module
from scripts.modeling import runtime as runtime_module


def _clear_runtime_caches() -> None:
    """Reset cached runtime probes between environment-sensitive tests."""
    runtime_module.cuda_visible.cache_clear()
    runtime_module._probe_xgboost_cuda.cache_clear()


def test_recommended_worker_cap_reserves_headroom(monkeypatch):
    """Leave a couple of cores free when auto-sizing workers from CPU count."""
    monkeypatch.delenv("ELF_MAX_WORKERS", raising=False)
    monkeypatch.delenv("ELF_RESERVED_CORES", raising=False)
    monkeypatch.setattr(runtime_module.os, "cpu_count", lambda: 24)

    assert runtime_module.recommended_worker_cap(16) == 16
    assert runtime_module.recommended_worker_cap(64) == 22


def test_recommended_worker_cap_honors_environment_override(monkeypatch):
    """Allow local runs to clamp worker count explicitly via the environment."""
    monkeypatch.setenv("ELF_MAX_WORKERS", "5")

    assert runtime_module.recommended_worker_cap(16) == 5


def test_recommended_inner_threads_per_worker_uses_safe_core_share(monkeypatch):
    """Clamp inner-thread budgets to a safe share of available host cores."""
    monkeypatch.delenv("ELF_INNER_THREADS_PER_WORKER", raising=False)
    monkeypatch.delenv("ELF_RESERVED_CORES", raising=False)
    monkeypatch.setattr(runtime_module.os, "cpu_count", lambda: 24)

    assert runtime_module.recommended_inner_threads_per_worker(4, outer_workers=10) == 2
    assert runtime_module.recommended_inner_threads_per_worker(4, outer_workers=4) == 4


def test_recommended_inner_threads_per_worker_honors_environment_override(monkeypatch):
    """Allow manual override of inner-thread budgets for local experimentation."""
    monkeypatch.setenv("ELF_INNER_THREADS_PER_WORKER", "3")

    assert runtime_module.recommended_inner_threads_per_worker(1, outer_workers=10) == 3


def test_supports_high_capacity_parallelism_detects_x64_hosts(monkeypatch):
    """Detect high-core x64 hosts that can use the boosted Stage-5 profile."""
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(runtime_module.os, "cpu_count", lambda: 24)

    assert runtime_module.supports_high_capacity_parallelism() is True


def test_supports_high_capacity_parallelism_rejects_arm_hosts(monkeypatch):
    """Keep ARM64 hosts on the conservative shared runtime profile."""
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(runtime_module.os, "cpu_count", lambda: 24)

    assert runtime_module.supports_high_capacity_parallelism() is False


def test_resolve_xgboost_runtime_respects_off_mode(monkeypatch):
    """Disable optional XGBoost catalog entries when acceleration is turned off."""
    _clear_runtime_caches()
    monkeypatch.setenv("ELF_ACCELERATION", "off")

    resolved = runtime_module.resolve_xgboost_runtime()

    assert resolved.available is False
    assert resolved.device is None
    assert resolved.reason == "disabled by ELF_ACCELERATION=off"


def test_resolve_xgboost_runtime_forces_cpu_when_requested(monkeypatch):
    """Keep optional XGBoost on CPU when the operator explicitly requests it."""
    _clear_runtime_caches()
    monkeypatch.setenv("ELF_ACCELERATION", "cpu")
    monkeypatch.setattr(runtime_module, "XGBRegressor", object)

    resolved = runtime_module.resolve_xgboost_runtime()

    assert resolved.available is True
    assert resolved.device == "cpu"
    assert resolved.cuda_enabled is False


def test_resolve_xgboost_runtime_survives_nvidia_smi_timeout(monkeypatch):
    """Keep acceleration probing opportunistic when nvidia-smi is temporarily slow."""
    _clear_runtime_caches()
    monkeypatch.setenv("ELF_ACCELERATION", "auto")

    class _FakeXGBRegressor:
        def __init__(self, **_kwargs):
            pass

        def fit(self, x_train, y_train):
            assert len(x_train) == len(y_train)
            return self

    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["nvidia-smi"], timeout=2)

    monkeypatch.setattr(runtime_module, "XGBRegressor", _FakeXGBRegressor)
    monkeypatch.setattr(runtime_module.subprocess, "run", _timeout)

    resolved = runtime_module.resolve_xgboost_runtime()

    assert resolved.available is True
    assert resolved.device == "cuda"
    assert resolved.cuda_enabled is True
    assert "timed out" in resolved.reason


def test_build_model_catalog_includes_optional_xgboost_specs(monkeypatch):
    """Expose XGBoost variants when the runtime helper says they are available."""
    monkeypatch.setattr(common_module, "XGBRegressor", object)
    monkeypatch.setattr(
        common_module,
        "resolve_xgboost_runtime",
        lambda: SimpleNamespace(
            available=True,
            device="cpu",
            cuda_enabled=False,
            reason="unit-test",
        ),
    )

    catalog = common_module.build_model_catalog(include_optional_xgb=True)

    assert "xgb-balanced" in catalog
    assert catalog["xgb-balanced"].family == "xgb"
    assert catalog["xgb-balanced"].params["device"] == "cpu"
