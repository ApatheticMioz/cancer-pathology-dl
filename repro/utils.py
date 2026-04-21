from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Iterable


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return str(datetime.timedelta(seconds=seconds))


def atomic_json_write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def count_files(directory: Path, patterns: list[str]) -> int:
    if not directory.exists():
        return 0
    total = 0
    for pattern in patterns:
        total += len(list(directory.rglob(pattern)))
    return total
