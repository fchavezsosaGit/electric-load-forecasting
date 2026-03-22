"""Tests for shared modeling parallel runtime helpers."""

from __future__ import annotations

from scripts.config import MODELING_PARALLEL
import scripts.modeling.parallel as parallel_module
from scripts.modeling.parallel import resolve_parallel_plan, run_stage_jobs


def _square(value: int) -> int:
    """Simple pure worker used to verify parallel helper behavior."""
    return value * value


def test_resolve_parallel_plan_enables_shared_runtime_for_large_batches():
    """Enable the shared parallel runtime once a stage exceeds the configured threshold."""
    plan = resolve_parallel_plan(
        "performance",
        task_count=max(2, int(MODELING_PARALLEL["min_tasks"])),
    )

    assert plan.stage == "performance"
    assert plan.task_count >= int(MODELING_PARALLEL["min_tasks"])
    assert plan.backend == MODELING_PARALLEL["backend"]
    assert plan.n_jobs >= 1


def test_run_stage_jobs_runs_sequentially_for_small_batches():
    """Keep execution sequential for very small task batches."""
    results, plan = run_stage_jobs("performance", [2], worker=_square)

    assert results == [4]
    assert plan.enabled is False
    assert plan.n_jobs == 1


def test_run_stage_jobs_returns_expected_results_for_parallel_batches():
    """Return correct results regardless of whether the helper chooses parallelism."""
    task_count = max(2, int(MODELING_PARALLEL["min_tasks"]))
    tasks = list(range(task_count))

    results, plan = run_stage_jobs("multires", tasks, worker=_square)

    assert results == [task * task for task in tasks]
    if plan.enabled:
        assert plan.n_jobs == min(plan.requested_workers, task_count)
    else:
        assert plan.n_jobs == 1


def test_resolve_parallel_plan_supports_rollout_and_forecast_control_stages():
    """Additional replay-heavy stages should share the centralized adaptive planner."""
    task_count = max(2, int(MODELING_PARALLEL["min_tasks"]))

    rollout_plan = resolve_parallel_plan("rollout_sweep", task_count=task_count)
    control_plan = resolve_parallel_plan("forecast_control", task_count=task_count)

    assert rollout_plan.stage == "rollout_sweep"
    assert rollout_plan.n_jobs >= 1
    assert control_plan.stage == "forecast_control"
    assert control_plan.n_jobs >= 1


def test_resolve_parallel_plan_uses_adaptive_worker_cap(monkeypatch):
    """Feed the resolved worker cap through into the shared runtime plan."""
    monkeypatch.setattr(parallel_module, "recommended_worker_cap", lambda _: 6)
    monkeypatch.setattr(parallel_module, "recommended_inner_threads_per_worker", lambda *_args, **_kwargs: 2)

    plan = resolve_parallel_plan(
        "multires",
        task_count=max(8, int(MODELING_PARALLEL["min_tasks"])),
    )

    assert plan.requested_workers == 6
    assert plan.n_jobs == 6
    assert plan.inner_threads_per_worker == 2


def test_resolve_parallel_plan_uses_high_capacity_stage_overrides(monkeypatch):
    """Use the boosted Stage-5 plan on high-capacity x64 hosts."""
    monkeypatch.setattr(parallel_module, "supports_high_capacity_parallelism", lambda: True)
    monkeypatch.setattr(parallel_module, "recommended_worker_cap", lambda configured: configured)
    monkeypatch.setattr(
        parallel_module,
        "recommended_inner_threads_per_worker",
        lambda configured, *, outer_workers: min(configured, outer_workers),
    )

    plan = resolve_parallel_plan("performance", task_count=80)

    assert plan.requested_workers == 10
    assert plan.n_jobs == 10
    assert plan.inner_threads_per_worker == 4


def test_resolve_parallel_plan_falls_back_to_global_limits_on_standard_hosts(monkeypatch):
    """Keep the conservative shared plan on non-boosted hosts."""
    monkeypatch.setattr(parallel_module, "supports_high_capacity_parallelism", lambda: False)
    monkeypatch.setattr(parallel_module, "recommended_worker_cap", lambda configured: configured)
    monkeypatch.setattr(
        parallel_module,
        "recommended_inner_threads_per_worker",
        lambda configured, *, outer_workers: configured + outer_workers,
    )

    plan = resolve_parallel_plan("performance", task_count=80)

    assert plan.requested_workers == int(MODELING_PARALLEL["max_workers"])
    assert plan.n_jobs == int(MODELING_PARALLEL["max_workers"])
    assert plan.inner_threads_per_worker == int(MODELING_PARALLEL["inner_threads_per_worker"]) + plan.n_jobs
