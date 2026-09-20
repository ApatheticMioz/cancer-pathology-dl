#!/usr/bin/env python3
"""Probe: splitter invariant hardening verification (one pass).

Verifies the hardened splitters in src/data.py:
  (a) synthetic declared-'patient' dataset with trivial groups  -> expect FATAL
  (b) synthetic declared-'patch'  dataset with trivial groups  -> expect OK, branch logged
  (c) real datasets (tcga, panda, siim, pannuke): parse + make_group_kfold_splits
      (n_splits=5, seed=42) -> report branch used + per-fold class coverage.

All detailed output is written to /tmp/split_probe.log. To inspect:
    grep -E 'PASS|FAIL|BRANCH' /tmp/split_probe.log | tail -30
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import DATASET_META, DATASET_ROOTS  # noqa: E402
from src.data import (  # noqa: E402
    load_dataset_bundle,
    make_group_kfold_splits,
)

LOG = Path("/tmp/split_probe.log")
_lines: list[str] = []


def log(msg: str) -> None:
    _lines.append(msg)


def flush() -> None:
    LOG.write_text("\n".join(_lines) + "\n")


def _run(fn, expect: str):
    """Run fn(); return (ok, detail). expect in {'fatal', 'ok'}."""
    try:
        res = fn()
        if expect == "fatal":
            return False, "expected FATAL but call succeeded"
        return True, res
    except Exception as e:  # noqa: BLE001
        if expect == "fatal":
            return True, f"FATAL as expected ({type(e).__name__})"
        return False, f"unexpected {type(e).__name__}: {e}"


def probe_a() -> None:
    # (a) declared 'patient' but trivial groups (one group per sample) -> FATAL
    n = 100
    labels = np.array([0] * 50 + [1] * 50)
    groups = np.arange(n)  # trivial: each sample its own group
    ok, detail = _run(
        lambda: make_group_kfold_splits(labels, groups, n_splits=5, seed=42, grouping="patient"),
        expect="fatal",
    )
    log(f"(a) synthetic patient+trivial: {'PASS' if ok else 'FAIL'} - {detail}")


def probe_b() -> None:
    # (b) declared 'patch' with trivial groups -> OK, branch logged
    n = 100
    labels = np.array([0] * 50 + [1] * 50)
    groups = np.arange(n)  # trivial
    ok, res = _run(
        lambda: make_group_kfold_splits(labels, groups, n_splits=5, seed=42, grouping="patch"),
        expect="ok",
    )
    if ok:
        log(f"(b) synthetic patch+trivial: PASS - BRANCH={res.metadata['branch']}")
    else:
        log(f"(b) synthetic patch+trivial: FAIL - {res}")


def _per_fold_coverage(splits, labels) -> tuple[bool, list[str]]:
    all_classes = sorted(int(c) for c in np.unique(labels))
    problems: list[str] = []
    for fold_idx, (_tr, vl) in enumerate(splits):
        val_counts = {c: 0 for c in all_classes}
        for lab in labels[vl]:
            val_counts[int(lab)] += 1
        missing = [c for c in all_classes if val_counts[c] == 0]
        if missing:
            problems.append(f"fold{fold_idx + 1} missing classes {missing}")
    return (not problems), problems


def probe_c() -> None:
    for ds in ["tcga", "panda", "siim", "pannuke"]:
        meta = DATASET_META[ds]
        grouping = meta["grouping"]
        # The grouping-provenance sidecar (if the builder emitted one) sits next
        # to the dataset index. Passing it lets a declared-'patient' dataset
        # with a legitimate one-image-per-patient release use the loud,
        # recorded trivial-patient branch instead of FATALing.
        provenance_path = DATASET_ROOTS[ds] / "preprocessed" / "grouping_provenance.json"
        try:
            bundle = load_dataset_bundle(ds, DATASET_ROOTS[ds], skip_macenko=False)
        except Exception as e:  # noqa: BLE001
            log(f"(c) {ds}: FAIL - parse error {type(e).__name__}: {e}")
            continue
        labels = bundle["labels"]
        groups = bundle["groups"]
        n = len(labels)
        n_unique = int(len(np.unique(groups)))
        log(f"(c) {ds}: n={n}, n_unique_groups={n_unique}, declared_grouping={grouping}")
        try:
            splits = make_group_kfold_splits(
                labels, groups, n_splits=5, seed=42, grouping=grouping,
                provenance_path=provenance_path,
            )
            branch = splits.metadata["branch"]
            log(f"(c) {ds}: BRANCH={branch}")
            ok, problems = _per_fold_coverage(splits, labels)
            if ok:
                n_cls = len(np.unique(labels))
                log(f"(c) {ds}: PASS - all 5 folds cover all {n_cls} classes")
            else:
                log(f"(c) {ds}: FAIL - per-fold class coverage: {'; '.join(problems)}")
        except Exception as e:  # noqa: BLE001
            # A declared-'patient' dataset whose on-disk groups are trivial AND
            # has no provenance sidecar proving patient-level grouping is the
            # exact latent bug this mutation is designed to catch -> FATAL.
            if grouping == "patient" and n_unique == n:
                log(f"(c) {ds}: FATAL (expected: declared patient, trivial groups, no provenance) - {type(e).__name__}")
            else:
                log(f"(c) {ds}: FAIL - unexpected {type(e).__name__}: {e}")


def main() -> int:
    probe_a()
    probe_b()
    probe_c()
    flush()
    print("probe complete; see /tmp/split_probe.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
