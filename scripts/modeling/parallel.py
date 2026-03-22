"""Shared joblib-backed execution helpers for modeling stages."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from typing import TypeVar

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from config import MODELING_PARALLEL, MODELING_STAGE_PARALLEL
from modeling.runtime import (
    recommended_inner_threads_per_worker,
    recommended_worker_cap,
    supports_high_capacity_parallelism,
)

TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParallelPlan:
    """Resolved execution plan for a batch of modeling jobs."""

    stage: str
    enabled: bool
    backend: str
    requested_workers: int
    n_jobs: int
    task_count: int
    batch_size: int
    pre_dispatch: str
    min_tasks: int
    inner_threads_per_worker: int

    def as_dict(self) -> dict[str, int | str | bool]:
        """Return a manifest-safe dictionary representation."""
        return asdict(self)


def resolve_parallel_plan(stage: str, *, task_count: int) -> ParallelPlan:
    """Resolve the effective execution strategy for a modeling stage."""
    stage_key = stage.strip().lower()
    if stage_key not in MODELING_STAGE_PARALLEL:
        raise ValueError(f"Unsupported modeling stage for parallel plan: {stage}")
    stage_config = MODELING_STAGE_PARALLEL[stage_key]
    configured_max_workers = int(stage_config["max_workers"])
    configured_inner_threads = int(stage_config["inner_threads_per_worker"])
    if bool(stage_config["high_capacity_host_only"]) and not supports_high_capacity_parallelism():
        configured_max_workers = int(MODELING_PARALLEL["max_workers"])
        configured_inner_threads = int(MODELING_PARALLEL["inner_threads_per_worker"])
    requested_workers = recommended_worker_cap(configured_max_workers)
    backend = str(MODELING_PARALLEL["backend"])
    enabled = (
        bool(MODELING_PARALLEL["enabled"])
        and bool(stage_config["enabled"])
        and backend != "sequential"
        and requested_workers > 1
        and int(task_count) >= int(MODELING_PARALLEL["min_tasks"])
    )
    n_jobs = min(requested_workers, max(1, int(task_count))) if enabled else 1
    inner_threads_per_worker = recommended_inner_threads_per_worker(
        configured_inner_threads,
        outer_workers=n_jobs,
    )
    return ParallelPlan(
        stage=stage_key,
        enabled=enabled,
        backend=backend,
        requested_workers=requested_workers,
        n_jobs=n_jobs,
        task_count=int(task_count),
        batch_size=int(MODELING_PARALLEL["batch_size"]),
        pre_dispatch=str(MODELING_PARALLEL["pre_dispatch"]),
        min_tasks=int(MODELING_PARALLEL["min_tasks"]),
        inner_threads_per_worker=inner_threads_per_worker,
    )


def _execute_with_thread_limits(
    task: TaskT,
    *,
    worker: Callable[[TaskT], ResultT],
    inner_threads_per_worker: int,
) -> ResultT:
    """Run one task while limiting nested math-library thread pools."""
    with threadpool_limits(limits=inner_threads_per_worker):
        return worker(task)


def run_stage_jobs(
    stage: str,
    tasks: Sequence[TaskT],
    *,
    worker: Callable[[TaskT], ResultT],
    logger_instance: logging.Logger | None = None,
) -> tuple[list[ResultT], ParallelPlan]:
    """Execute modeling jobs sequentially or via joblib from shared config."""
    plan = resolve_parallel_plan(stage, task_count=len(tasks))
    if not tasks:
        return [], plan

    active_logger = logger_instance or logger
    if plan.enabled:
        active_logger.info(
            "Parallelizing %s jobs: stage=%s backend=%s tasks=%d workers=%d",
            plan.task_count,
            plan.stage,
            plan.backend,
            plan.task_count,
            plan.n_jobs,
        )
        wrapped_worker = partial(
            _execute_with_thread_limits,
            worker=worker,
            inner_threads_per_worker=plan.inner_threads_per_worker,
        )
        results = Parallel(
            n_jobs=plan.n_jobs,
            backend=plan.backend,
            batch_size=plan.batch_size,
            pre_dispatch=plan.pre_dispatch,
        )(delayed(wrapped_worker)(task) for task in tasks)
        return list(results), plan

    active_logger.info(
        "Running %s jobs sequentially: stage=%s backend=%s tasks=%d",
        plan.task_count,
        plan.stage,
        plan.backend,
        plan.task_count,
    )
    results = [
        _execute_with_thread_limits(
            task,
            worker=worker,
            inner_threads_per_worker=plan.inner_threads_per_worker,
        )
        for task in tasks
    ]
    return results, plan
