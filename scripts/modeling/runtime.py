"""Runtime hardware helpers for adaptive worker sizing and optional acceleration."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_RESERVED_CORES = 2
HIGH_CAPACITY_CPU_THRESHOLD = 16
VALID_ACCELERATION_MODES = {"auto", "cpu", "gpu", "off"}

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - exercised via runtime fallback tests.
    XGBRegressor = None


@dataclass(frozen=True)
class XGBoostRuntime:
    """Resolved optional XGBoost runtime contract for this process."""

    available: bool
    device: str | None
    cuda_enabled: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return a manifest-safe dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class RuntimeSummary:
    """Compact runtime summary emitted into manifests and diagnostics."""

    machine: str
    cpu_count: int
    worker_cap: int
    acceleration_mode: str
    xgboost: XGBoostRuntime

    def as_dict(self) -> dict[str, Any]:
        """Return a manifest-safe dictionary representation."""
        payload = asdict(self)
        payload["xgboost"] = self.xgboost.as_dict()
        return payload


def acceleration_mode() -> str:
    """Resolve the requested acceleration policy from the environment."""
    raw_value = os.getenv("ELF_ACCELERATION", "auto").strip().lower()
    if raw_value not in VALID_ACCELERATION_MODES:
        logger.warning(
            "Unsupported ELF_ACCELERATION=%r; falling back to 'auto'. Supported: %s",
            raw_value,
            sorted(VALID_ACCELERATION_MODES),
        )
        return "auto"
    return raw_value


def _env_positive_int(name: str) -> int | None:
    """Read one positive integer override from the environment when present."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Ignoring %s=%r because it is not an integer.", name, raw_value)
        return None
    if value <= 0:
        logger.warning("Ignoring %s=%r because it must be positive.", name, raw_value)
        return None
    return value


def recommended_worker_cap(configured_max_workers: int) -> int:
    """Resolve the effective outer-worker cap for the current machine."""
    override = _env_positive_int("ELF_MAX_WORKERS")
    if override is not None:
        return override

    cpu_count = max(1, os.cpu_count() or 1)
    reserved_cores = _env_positive_int("ELF_RESERVED_CORES") or DEFAULT_RESERVED_CORES
    auto_cap = max(1, cpu_count - max(0, reserved_cores))
    return max(1, min(int(configured_max_workers), auto_cap))


def supports_high_capacity_parallelism() -> bool:
    """Return whether this host should opt into boosted x64 parallel plans."""
    machine = platform.machine().strip().lower()
    return machine in {"amd64", "x86_64"} and max(1, os.cpu_count() or 1) >= HIGH_CAPACITY_CPU_THRESHOLD


def recommended_inner_threads_per_worker(configured_inner_threads: int, *, outer_workers: int) -> int:
    """Resolve a safe inner-thread budget for each parallel worker."""
    override = _env_positive_int("ELF_INNER_THREADS_PER_WORKER")
    if override is not None:
        return override
    cpu_count = max(1, os.cpu_count() or 1)
    reserved_cores = _env_positive_int("ELF_RESERVED_CORES") or DEFAULT_RESERVED_CORES
    available_cores = max(1, cpu_count - max(0, reserved_cores))
    safe_share = max(1, available_cores // max(1, int(outer_workers)))
    return max(1, min(int(configured_inner_threads), safe_share))


@lru_cache(maxsize=1)
def cuda_visible() -> bool | None:
    """Return whether an NVIDIA CUDA device appears reachable from this process."""
    if acceleration_mode() in {"cpu", "off"}:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except OSError:
        return False
    except subprocess.TimeoutExpired:
        return None
    return result.returncode == 0 and bool(result.stdout.strip())


@lru_cache(maxsize=1)
def _probe_xgboost_cuda() -> tuple[bool, str]:
    """Probe whether the installed XGBoost build can train on CUDA."""
    if XGBRegressor is None:
        return False, "xgboost is not installed"
    cuda_report = cuda_visible()
    if cuda_report is False:
        return False, "nvidia-smi did not report a CUDA device"
    preface = ""
    if cuda_report is None:
        preface = "nvidia-smi probe timed out; falling back to direct XGBoost CUDA probe. "
    try:
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            n_estimators=1,
            max_depth=1,
            learning_rate=1.0,
            n_jobs=1,
            random_state=42,
            verbosity=0,
        )
        x_train = np.arange(8, dtype=float).reshape(-1, 1)
        y_train = np.arange(8, dtype=float)
        model.fit(x_train, y_train)
    except Exception as exc:  # pragma: no cover - depends on local accelerator stack.
        return False, f"{preface}CUDA probe failed: {exc.__class__.__name__}".strip()
    return True, f"{preface}CUDA probe fit succeeded".strip()


def resolve_xgboost_runtime() -> XGBoostRuntime:
    """Resolve whether optional XGBoost models should be added to the catalog."""
    mode = acceleration_mode()
    if mode == "off":
        return XGBoostRuntime(
            available=False,
            device=None,
            cuda_enabled=False,
            reason="disabled by ELF_ACCELERATION=off",
        )
    if XGBRegressor is None:
        return XGBoostRuntime(
            available=False,
            device=None,
            cuda_enabled=False,
            reason="xgboost is not installed",
        )
    if mode == "cpu":
        return XGBoostRuntime(
            available=True,
            device="cpu",
            cuda_enabled=False,
            reason="forced to CPU by ELF_ACCELERATION=cpu",
        )

    cuda_ready, reason = _probe_xgboost_cuda()
    if cuda_ready:
        return XGBoostRuntime(
            available=True,
            device="cuda",
            cuda_enabled=True,
            reason=reason,
        )
    return XGBoostRuntime(
        available=True,
        device="cpu",
        cuda_enabled=False,
        reason=reason,
    )


def runtime_summary(configured_max_workers: int) -> RuntimeSummary:
    """Return a compact snapshot of the local runtime surface."""
    return RuntimeSummary(
        machine=platform.machine(),
        cpu_count=max(1, os.cpu_count() or 1),
        worker_cap=recommended_worker_cap(configured_max_workers),
        acceleration_mode=acceleration_mode(),
        xgboost=resolve_xgboost_runtime(),
    )
