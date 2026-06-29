"""DataLoader tuning logic: batch size, cache, and worker selection.

Provides:
    - _logical_cpu_count: Number of logical CPUs available.
    - _available_ram_gb: Available RAM in GB.
    - resolve_batch_size: Resolve effective batch size.
    - _select_cache_size: Choose dataset cache size.
    - _initial_loader_tuning: Choose workers, prefetch, persistent settings.
"""
from __future__ import annotations

import os
from pathlib import Path

from src.config import DEFAULT_BATCH_SIZE


def _logical_cpu_count() -> int:
    """Return the number of logical CPUs available to this process."""
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return max(1, len(affinity))
    except Exception:
        pass
    return max(1, os.cpu_count() or 8)


def _available_ram_gb() -> float:
    """Return available RAM in GB (Linux /proc/meminfo)."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return 0.0
    available_kb = None
    total_kb = None
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available_kb = float(line.split()[1])
            elif line.startswith("MemTotal:"):
                total_kb = float(line.split()[1])
        value_kb = available_kb if available_kb is not None else total_kb
        if value_kb is None:
            return 0.0
        return float(value_kb) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def resolve_batch_size(requested: int | None) -> int:
    if requested and int(requested) > 0:
        return int(requested)
    return int(DEFAULT_BATCH_SIZE)


def _select_cache_size(dataset: str, requested_cache: int, available_ram_gb: float, workers: int, allow_big_cache: bool) -> int:
    if requested_cache == 0:
        return 0
    if requested_cache > 0:
        if dataset in {"panda", "siim", "pannuke"} and not allow_big_cache:
            return 0
        return int(requested_cache)
    if dataset == "tcga" and available_ram_gb >= 18.0 and workers > 0:
        return 48 if dataset == "tcga" else 64
    return 0


def _initial_loader_tuning(dataset_name: str, requested_workers: int, available_ram_gb: float, cpu_budget: int):
    """Return (num_workers, prefetch_factor, persistent_workers)."""
    requested_workers = int(requested_workers)
    cpu_budget = max(1, int(cpu_budget))
    if requested_workers == 0:
        return 0, 0, False
    if requested_workers < 0:
        if dataset_name == "tcga":
            target = 12 if available_ram_gb >= 20.0 else 8
            workers = min(cpu_budget, target)
            return workers, 4, True
        if dataset_name in {"panda", "siim", "pannuke"}:
            target = 10 if available_ram_gb >= 24.0 else 8
            workers = min(cpu_budget, target)
            return workers, 2, False
        workers = min(cpu_budget, 8)
        return workers, 2, True
    if dataset_name == "tcga":
        workers = min(requested_workers, min(cpu_budget, 12))
        return workers, 4, True
    if dataset_name in {"panda", "siim", "pannuke"}:
        workers = min(requested_workers, min(cpu_budget, 10))
        return workers, 2, False
    workers = min(requested_workers, min(cpu_budget, 8))
    return workers, 2, True