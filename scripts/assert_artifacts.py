#!/usr/bin/env python3
"""R3-1 artifact gate: standalone checker for ``results/kfold_campaign/<run_label>/``.

Validates the deterministic artifact contract produced by
``main.py --run-label <label>`` (see src/training.py F-10 / F-23):

  * ``epoch_log.jsonl``    -- non-empty JSONL; every record carries
                              run_label / fold / seed / splitter_branch
                              with correct types (splitter_branch non-empty).
  * ``per_slice_dice.jsonl`` -- non-empty JSONL; every record carries
                              EXACTLY the 10 fields
                              {run_label, dataset, encoder, fold, seed,
                               case_id, dice, empty_pred, empty_gt, label_int}
                              with correct types.
  * ``best.pt``            -- loadable torch state dict of tensors.
  * ``final.state.pt``     -- loadable training state with epoch (int),
                              model_state (dict), optimizer_state,
                              fingerprint (dict).
  * summary JSON (optional, ``--summary``) -- every run entry (or every
                              fold entry under ``fold_results``) carries
                              ``resume_branch`` (str) and
                              ``resumed_from_epoch`` (int); optional
                              ``--expect-resumed-from-epoch`` /
                              ``--expect-resume-branch`` equality asserts.

Standalone-checkable: ``--results-dir`` overrides the ``results/kfold_campaign``
root so the checker can be pointed at a /tmp fixture directory.

Exit status: 0 = all artifacts valid; 1 = FATAL (missing / corrupt /
mistyped artifact). Never prompts; safe for unattended use.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "kfold_campaign"

# ---------------------------------------------------------------------------
# Field contracts
# ---------------------------------------------------------------------------
EPOCH_LOG_FIELDS: dict[str, type] = {
    "run_label": str,
    "fold": int,
    "seed": int,
    "splitter_branch": str,
}

PER_SLICE_FIELDS: dict[str, type] = {
    "run_label": str,
    "dataset": str,
    "encoder": str,
    "fold": int,
    "seed": int,
    "case_id": str,
    "dice": float,
    "empty_pred": bool,
    "empty_gt": bool,
    "label_int": int,
}

STATE_REQUIRED_KEYS = ("epoch", "model_state", "optimizer_state", "fingerprint")


def fatal(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def _check_type(value, expected: type, ctx: str) -> None:
    if expected is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif expected is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    else:
        ok = isinstance(value, expected)
    if not ok:
        fatal(f"{ctx}: expected {expected.__name__}, got {type(value).__name__} ({value!r})")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        fatal(f"missing artifact: {path}")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        fatal(f"artifact is empty (0 records): {path}")
    records = []
    for i, line in enumerate(lines, 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as ex:
            fatal(f"{path} line {i}: corrupt JSON: {ex}")
    return records


# ---------------------------------------------------------------------------
# Individual artifact checks
# ---------------------------------------------------------------------------
def check_epoch_log(run_dir: Path, run_label: str) -> None:
    path = run_dir / "epoch_log.jsonl"
    records = _read_jsonl(path)
    for i, rec in enumerate(records, 1):
        ctx = f"{path} line {i}"
        for field, typ in EPOCH_LOG_FIELDS.items():
            if field not in rec:
                fatal(f"{ctx}: missing field '{field}'")
            _check_type(rec[field], typ, f"{ctx} field '{field}'")
        if rec["run_label"] != run_label:
            fatal(f"{ctx}: run_label {rec['run_label']!r} != expected {run_label!r}")
        if not rec["splitter_branch"]:
            fatal(f"{ctx}: splitter_branch is empty")
    print(f"OK: {path} ({len(records)} records; run_label/fold/seed/splitter_branch valid)")


def check_per_slice_dice(run_dir: Path, run_label: str) -> None:
    path = run_dir / "per_slice_dice.jsonl"
    records = _read_jsonl(path)
    expected_keys = set(PER_SLICE_FIELDS)
    for i, rec in enumerate(records, 1):
        ctx = f"{path} line {i}"
        if set(rec) != expected_keys:
            missing = expected_keys - set(rec)
            extra = set(rec) - expected_keys
            fatal(f"{ctx}: expected exactly {len(expected_keys)} fields {sorted(expected_keys)}; "
                  f"missing={sorted(missing)} extra={sorted(extra)}")
        for field, typ in PER_SLICE_FIELDS.items():
            _check_type(rec[field], typ, f"{ctx} field '{field}'")
        if rec["run_label"] != run_label:
            fatal(f"{ctx}: run_label {rec['run_label']!r} != expected {run_label!r}")
    print(f"OK: {path} ({len(records)} records; 10-field contract valid)")


def check_best_pt(run_dir: Path) -> None:
    path = run_dir / "best.pt"
    if not path.exists():
        fatal(f"missing artifact: {path}")
    try:
        state = torch.load(path, map_location="cpu")
    except Exception as ex:
        fatal(f"{path}: corrupt / unloadable: {ex!r}")
    if not isinstance(state, dict) or not state:
        fatal(f"{path}: not a non-empty state dict (got {type(state).__name__})")
    for key, val in state.items():
        if not torch.is_tensor(val):
            fatal(f"{path}: key {key!r} is not a tensor (got {type(val).__name__})")
    print(f"OK: {path} ({len(state)} tensors)")


def check_state_pt(run_dir: Path) -> None:
    path = run_dir / "final.state.pt"
    if not path.exists():
        fatal(f"missing artifact: {path}")
    try:
        state = torch.load(path, map_location="cpu")
    except Exception as ex:
        fatal(f"{path}: corrupt / unloadable: {ex!r}")
    if not isinstance(state, dict):
        fatal(f"{path}: not a dict (got {type(state).__name__})")
    for key in STATE_REQUIRED_KEYS:
        if key not in state:
            fatal(f"{path}: missing key '{key}'")
    _check_type(state["epoch"], int, f"{path} key 'epoch'")
    if not isinstance(state["model_state"], dict) or not state["model_state"]:
        fatal(f"{path}: 'model_state' is not a non-empty dict")
    if not isinstance(state["fingerprint"], dict):
        fatal(f"{path}: 'fingerprint' is not a dict")
    print(f"OK: {path} (epoch={state['epoch']}, fingerprint keys={sorted(state['fingerprint'])})")


def check_summary(summary_path: Path,
                 expect_resumed: int | None,
                 expect_branch: str | None) -> None:
    if not summary_path.exists():
        fatal(f"missing summary: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as ex:
        fatal(f"{summary_path}: corrupt JSON: {ex}")
    runs = summary.get("runs")
    if not isinstance(runs, dict) or not runs:
        fatal(f"{summary_path}: no non-empty 'runs' dict")
    n_checked = 0
    for key, entry in runs.items():
        # k-fold CV summaries nest per-fold results; single-split is flat.
        fold_results = entry.get("fold_results")
        entries = fold_results if isinstance(fold_results, list) and fold_results else [entry]
        for e in entries:
            ctx = f"{summary_path} runs[{key!r}]"
            for field in ("resume_branch", "resumed_from_epoch"):
                if field not in e:
                    fatal(f"{ctx}: missing '{field}'")
            _check_type(e["resume_branch"], str, f"{ctx} field 'resume_branch'")
            _check_type(e["resumed_from_epoch"], int, f"{ctx} field 'resumed_from_epoch'")
            if expect_branch is not None and e["resume_branch"] != expect_branch:
                fatal(f"{ctx}: resume_branch {e['resume_branch']!r} != expected {expect_branch!r}")
            if expect_resumed is not None and e["resumed_from_epoch"] != expect_resumed:
                fatal(f"{ctx}: resumed_from_epoch {e['resumed_from_epoch']} != expected {expect_resumed}")
            n_checked += 1
    print(f"OK: {summary_path} ({n_checked} run/fold entries carry resume_branch + resumed_from_epoch)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="R3-1 artifact gate (standalone)")
    p.add_argument("--run-label", required=True, help="run label under results/kfold_campaign/")
    p.add_argument("--summary", default=None, help="optional summary JSON to validate")
    p.add_argument("--expect-resumed-from-epoch", type=int, default=None,
                   help="assert every entry's resumed_from_epoch equals this value")
    p.add_argument("--expect-resume-branch", default=None,
                   help="assert every entry's resume_branch equals this value")
    p.add_argument("--results-dir", default=None,
                   help="override results/kfold_campaign root (e.g. a /tmp fixture dir)")
    args = p.parse_args()

    base = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR
    run_dir = base / args.run_label
    if not run_dir.is_dir():
        fatal(f"run directory missing: {run_dir}")

    check_epoch_log(run_dir, args.run_label)
    check_per_slice_dice(run_dir, args.run_label)
    check_best_pt(run_dir)
    check_state_pt(run_dir)
    if args.summary:
        check_summary(Path(args.summary), args.expect_resumed_from_epoch, args.expect_resume_branch)

    print(f"PASS: all artifacts valid for run-label '{args.run_label}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
