"""Atomic file I/O and utility helpers for audit logging.

Provides crash-safe JSON/JSONL writing, timestamp formatting,
and basic filesystem helpers used across the training pipeline.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def now_iso() -> str:
    """Return the current UTC-local timestamp in ISO-8601 format."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def fmt_seconds(seconds: float) -> str:
    """Format a float number of seconds as a human-readable HH:MM:SS string."""
    seconds = max(0, int(seconds))
    return str(datetime.timedelta(seconds=seconds))


def atomic_json_write(path: Path, obj: dict) -> None:
    """Write *obj* to *path* atomically via a temporary file + rename.

    Guarantees that partial writes never corrupt the target file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)
    logger.debug("Atomic JSON write: %s", path)


def append_jsonl(path: Path, record: dict) -> None:
    """Append a single JSON record to a JSONL log file.

    Creates parent directories if they do not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def first_existing(paths: Iterable[Path]) -> Path | None:
    """Return the first path in *paths* that exists on disk, or ``None``."""
    for p in paths:
        if p.exists():
            return p
    return None


def count_files(directory: Path, patterns: list[str]) -> int:
    """Recursively count files in *directory* matching any of *patterns*."""
    if not directory.exists():
        return 0
    total = 0
    for pattern in patterns:
        total += len(list(directory.rglob(pattern)))
    return total