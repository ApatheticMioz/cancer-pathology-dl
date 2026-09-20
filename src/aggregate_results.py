#!/usr/bin/env python3
"""Results aggregator for the experimental matrix.

Scans all JSON summary files matching checkpoints/summary_*.json,
validates all 26 expected runs, extracts per-run metrics, and exports:
  - paper/paper_results_matrix.csv       (full results table)
  - paper/paper_results_latex_table.txt  (booktabs-ready LaTeX)

Usage:
    python src/aggregate_results.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"
PAPER_DIR = BASE_DIR / "paper"
RESULTS_DIR = BASE_DIR / "results"
ROUND2_DIR = RESULTS_DIR / "round2"
CSV_OUTPUT = PAPER_DIR / "paper_results_matrix.csv"
LATEX_OUTPUT = PAPER_DIR / "paper_results_latex_table.txt"
# New CI-augmented matrix (kfold mean±SD + Wilson Acc CIs + bootstrap Dice CIs).
CSV_WITH_CI_OUTPUT = PAPER_DIR / "paper_results_matrix_with_ci.csv"

# ---------------------------------------------------------------------------
# Expected 26-run experimental matrix (from run_all_experiments.sh)
# ---------------------------------------------------------------------------
EXPECTED_RUNS = [
    # Group 1: The Naked Baseline (6 runs)
    ("01", "g1_tcga_vgg16"),
    ("02", "g1_tcga_mobilenet_v2"),
    ("03", "g1_panda_vgg16"),
    ("04", "g1_panda_mobilenet_v2"),
    ("05", "g1_siim_vgg16"),
    ("06", "g1_siim_mobilenet_v2"),
    # Group 2: The Final Form Package (6 runs)
    ("07", "g2_tcga_vgg16"),
    ("08", "g2_tcga_mobilenet_v2"),
    ("09", "g2_panda_vgg16"),
    ("10", "g2_panda_mobilenet_v2"),
    ("11", "g2_siim_vgg16"),
    ("12", "g2_siim_mobilenet_v2"),
    # Group 3: The PanNuke Crucible (4 runs)
    ("13", "g3_pannuke_vgg16_naked"),
    ("14", "g3_pannuke_mobilenet_v2_naked"),
    ("15", "g3_pannuke_vgg16_final"),
    ("16", "g3_pannuke_mobilenet_v2_final"),
    # Group 4: The Optimization Teardown (6 runs)
    ("17", "g4_panda_isolate_lr"),
    ("18", "g4_panda_isolate_gn"),
    ("19", "g4_panda_lambda_1_1"),
    ("20", "g4_panda_lambda_5_1"),
    ("21", "g4_panda_lambda_1_10"),
    ("22", "g4_panda_lambda_10_1"),
    # Group 5: Preprocessing & Architecture Ablations (4 runs)
    ("23", "g5_panda_nomacenko"),
    ("24", "g5_pannuke_nomacenko"),
    ("25", "g5_tcga_noskip"),
    ("26", "g5_panda_noskip"),
]

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "Run Label",
    "Phase",
    "Dataset",
    "Encoder",
    "LR",
    "Use GradNorm",
    "GradNorm Alpha",
    "Skip Connections",
    "Macenko",
    "Seg Weight",
    "Cls Weight",
    "Lambda Ratio (Seg:Cls)",
    "Accuracy (%)",
    "Macro Dice (%)",
    "Paper Acc (%)",
    "Paper Dice (%)",
    "Acc Delta (%)",
    "Dice Delta (%)",
    "Status",
    "Timestamp",
    # --- Round-2 CI augmentation (kfold CV + bootstrap/Wilson) ---
    "Kfold Acc Mean (%)",
    "Kfold Acc SD (%)",
    "Kfold Dice Mean (%)",
    "Kfold Dice SD (%)",
    "Kfold N",
    "Acc 95% CI Lower",
    "Acc 95% CI Upper",
    "Acc 95% CI",
    "Dice 95% CI Lower (bootstrap)",
    "Dice 95% CI Upper (bootstrap)",
    "Dice 95% CI (bootstrap)",
    "Dice 95% CI Lower (pos-only)",
    "Dice 95% CI Upper (pos-only)",
    "Dice 95% CI Lower (neg-only)",
    "Dice 95% CI Upper (neg-only)",
    "Splitter Branch",
]


def discover_summary_files() -> list[Path]:
    """Find all per-run summary JSON files in checkpoints/.

    Matches:
      - summary_*.json   (individual run summaries from the concurrent orchestrator)
    """
    files: list[Path] = []

    files.extend(CHECKPOINT_DIR.glob("summary_*.json"))

    files = sorted(set(files), key=lambda p: p.stat().st_mtime)
    return files


def find_matching_log(summary_path: Path) -> Path | None:
    """Find the log file that corresponds to a summary file.

    Summary files are named: summary_XX_name.json
    Log files are named:     run_XX_name.log

    Falls back to matching by run ID prefix if exact name doesn't match.
    """
    stem = summary_path.stem  # e.g. "summary_09_g2_panda_vgg16"
    parts = stem.split("_", 2)  # ["summary", "09", "g2_panda_vgg16"]
    if len(parts) < 3:
        return None

    run_id = parts[1]
    run_name = parts[2]

    exact = LOGS_DIR / f"run_{run_id}_{run_name}.log"
    if exact.exists():
        return exact

    for log_file in sorted(LOGS_DIR.glob(f"run_{run_id}_*.log")):
        return log_file

    return None


def extract_lr_from_log(log_path: Path) -> str:
    """Extract learning rate from a run log file.

    Log lines contain: 'Phase v2 applied: lr=0.0001, ...'
    Returns the LR as a string (e.g. "1e-4" or "0.001").
    """
    lr_pattern = re.compile(r"Phase \S+ applied:\s+lr=([\d.eE+-]+)")
    try:
        text = log_path.read_text()
    except OSError:
        return ""

    matches = lr_pattern.findall(text)
    if not matches:
        return ""

    last_lr = matches[-1]
    lr_val = float(last_lr)
    if lr_val == 1e-3:
        return "1e-3"
    elif lr_val == 1e-4:
        return "1e-4"
    elif lr_val == 1e-2:
        return "1e-2"
    elif lr_val == 1e-1:
        return "1e-1"
    else:
        return f"{lr_val:.0e}"


def validate_all_runs() -> tuple[bool, list[str], dict[str, str]]:
    """Validate that all 26 expected runs have valid summary files.

    Returns:
        (all_valid, list_of_issues, lr_map)
        - all_valid: True only if all 26 runs pass validation
        - list_of_issues: human-readable issue descriptions
        - lr_map: dict mapping summary filename -> LR string
    """
    issues: list[str] = []
    lr_map: dict[str, str] = {}
    valid_count = 0

    for run_id, run_name in EXPECTED_RUNS:
        summary_file = CHECKPOINT_DIR / f"summary_{run_id}_{run_name}.json"
        label = f"Run {run_id.zfill(2)} ({run_name})"

        if not summary_file.exists():
            issues.append(f"  [FAIL] {label}: summary file MISSING ({summary_file.name})")
            log_path = find_matching_log(summary_file)
            if log_path:
                text = log_path.read_text()
                if "Traceback" in text or "Error" in text or "error" in text:
                    issues.append(f"         Log shows error/crash in {log_path.name}")
                elif "Final metrics" in text:
                    issues.append(f"         Log shows completion but summary was not written")
                else:
                    issues.append(f"         Log exists ({log_path.name}) but no clear completion")
            continue

        try:
            with open(summary_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            issues.append(f"  [FAIL] {label}: invalid JSON ({exc})")
            continue

        status = data.get("status", "")
        if status != "completed":
            issues.append(f"  [FAIL] {label}: status is '{status}' (expected 'completed')")
            continue

        runs = data.get("runs", {})
        if not runs:
            issues.append(f"  [FAIL] {label}: no run data in summary")
            continue

        for run_key, run_data in runs.items():
            acc = run_data.get("final_val_acc")
            dice = run_data.get("final_val_dice")
            if acc is None or dice is None:
                issues.append(f"  [FAIL] {label}: missing metrics for {run_key}")
                continue
            if not (0.0 <= acc <= 1.0 and 0.0 <= dice <= 1.0):
                issues.append(f"  [FAIL] {label}: out-of-range metrics acc={acc}, dice={dice}")
                continue

        log_path = find_matching_log(summary_file)
        if log_path:
            lr = extract_lr_from_log(log_path)
            lr_map[summary_file.name] = lr

        valid_count += 1
        print(f"  [OK]   {label}: status=completed, runs={list(runs.keys())}")

    failed_count = len(EXPECTED_RUNS) - valid_count
    all_valid = valid_count == len(EXPECTED_RUNS)
    return all_valid, issues, lr_map, valid_count, failed_count


def _lambda_ratio(seg: float | None, cls: float | None) -> str:
    """Format the seg:cls lambda ratio as a human-readable string."""
    if seg is None or cls is None or cls == 0:
        return ""
    import math
    g = math.gcd(int(round(seg * 100)), int(round(cls * 100)))
    s = int(round(seg * 100)) // g
    c = int(round(cls * 100)) // g
    return f"{s}:{c}"


def parse_summary_file(filepath: Path, lr: str = "") -> list[dict]:
    """Parse a single summary JSON and return a list of per-run records."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [WARN] Skipping {filepath.name}: {exc}", file=sys.stderr)
        return []

    phase = data.get("phase", "unknown")
    args = data.get("args", {})
    timestamp = data.get("timestamp", "")
    status = data.get("status", "")

    stem = filepath.stem
    parts = stem.split("_", 2)
    run_id = parts[1] if len(parts) >= 3 and parts[0] == "summary" and parts[1].isdigit() else ""
    run_name = parts[2] if len(parts) >= 3 and parts[0] == "summary" and parts[1].isdigit() else stem

    runs = data.get("runs", {})
    if not runs:
        print(f"  [WARN] No runs found in {filepath.name}", file=sys.stderr)
        return []

    # Known encoder names (order matters: longest first to avoid partial matches)
    _KNOWN_ENCODERS = ("mobilenet_v2", "vgg16")

    def _split_run_key(key: str) -> tuple[str, str]:
        for enc in _KNOWN_ENCODERS:
            if key.endswith(f"_{enc}"):
                return key[: -len(enc) - 1], enc
        return key, "unknown"

    records: list[dict] = []
    for run_key, run_data in runs.items():
        dataset, encoder = _split_run_key(run_key)

        acc = round(100.0 * run_data.get("final_val_acc", 0.0), 2)
        dice = round(100.0 * run_data.get("final_val_dice", 0.0), 2)

        paper_acc = run_data.get("paper_acc")
        paper_dice = run_data.get("paper_dice")
        acc_delta = run_data.get("acc_delta")
        dice_delta = run_data.get("dice_delta")

        paper_acc_str = f"{round(100.0 * paper_acc, 2)}" if paper_acc is not None else ""
        paper_dice_str = f"{round(100.0 * paper_dice, 2)}" if paper_dice is not None else ""
        acc_delta_str = f"{round(100.0 * acc_delta, 2)}" if acc_delta is not None else ""
        dice_delta_str = f"{round(100.0 * dice_delta, 2)}" if dice_delta is not None else ""

        lambda_seg = args.get("lambda_seg", "")
        lambda_cls = args.get("lambda_cls", "")
        gradnorm_alpha = args.get("gradnorm_alpha", "")
        no_skip = args.get("no_skip_connections", False)
        no_macenko = args.get("no_macenko", False)

        ratio_str = ""
        if isinstance(lambda_seg, (int, float)) and isinstance(lambda_cls, (int, float)) and lambda_cls != 0:
            ratio_str = _lambda_ratio(float(lambda_seg), float(lambda_cls))

        label_parts = [dataset.upper(), encoder]
        if no_skip:
            label_parts.append("no-skip")
        if no_macenko:
            label_parts.append("no-macenko")
        if gradnorm_alpha and gradnorm_alpha != 1.0:
            label_parts.append(f"alpha={gradnorm_alpha}")
        if ratio_str and ratio_str != "5:1":
            label_parts.append(f"lambda={ratio_str}")
        run_label = "-".join(label_parts)

        records.append({
            "Run Label": run_label,
            "Phase": phase,
            "Dataset": dataset.upper(),
            "Encoder": encoder,
            "LR": lr,
            "Use GradNorm": str(args.get("use_gradnorm", False)),
            "GradNorm Alpha": str(gradnorm_alpha) if gradnorm_alpha != "" else "",
            "Skip Connections": str(not no_skip),
            "Macenko": str(not no_macenko),
            "Seg Weight": str(lambda_seg) if lambda_seg != "" else "",
            "Cls Weight": str(lambda_cls) if lambda_cls != "" else "",
            "Lambda Ratio (Seg:Cls)": ratio_str,
            "Accuracy (%)": acc,
            "Macro Dice (%)": dice,
            "Paper Acc (%)": paper_acc_str,
            "Paper Dice (%)": paper_dice_str,
            "Acc Delta (%)": acc_delta_str,
            "Dice Delta (%)": dice_delta_str,
            "Status": status,
            "Timestamp": timestamp,
            "_run_id": run_id,
            "_run_name": run_name,
        })

    return records


def deduplicate_records(records: list[dict]) -> list[dict]:
    """Keep the latest record for each unique run.

    Uses _run_id (if available) to preserve all 26 distinct experimental runs,
    otherwise falls back to a complete configuration key including Use GradNorm.
    """
    seen: dict[tuple, dict] = {}
    for rec in records:
        if rec.get("_run_id"):
            key = (rec["_run_id"], rec["Dataset"], rec["Encoder"])
        else:
            key = (
                rec["Phase"],
                rec["Dataset"],
                rec["Encoder"],
                rec["LR"],
                rec["Use GradNorm"],
                rec["Skip Connections"],
                rec["Macenko"],
                rec["Seg Weight"],
                rec["Cls Weight"],
                rec["GradNorm Alpha"],
            )
        seen[key] = rec

    result = list(seen.values())
    result.sort(key=lambda r: (r.get("_run_id", "99"), r["Dataset"], r["Encoder"]))
    return result


# ---------------------------------------------------------------------------
# Round-2 CI augmentation: kfold CV mean±SD + Wilson Acc CIs + bootstrap Dice CIs
# ---------------------------------------------------------------------------
# Wilson CIs are strictly Acc-only (reused from scripts/compute_wilson_ci.py);
# Dice CIs are strictly percentile-bootstrap (reused from
# scripts/bootstrap_dice_ci.py).  The two are NEVER blurred.

# Exact per-dataset validation-set sizes (pinned from run logs; mirrors
# scripts/compute_wilson_ci.py::DATASET_VAL_SIZES) for single-split Wilson CIs.
DATASET_VAL_SIZES = {
    "TCGA": 778,
    "PANDA": 2104,
    "SIIM": 2135,
    "PANNUKE": 1567,
}

_BOOTSTRAP_MOD = None
_WILSON_MOD = None


def _load_script_module(name: str):
    """Import a scripts/*.py file by path (they are not a package)."""
    path = BASE_DIR / "scripts" / name
    mod_name = "agg_" + name.replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bootstrap_mod():
    global _BOOTSTRAP_MOD
    if _BOOTSTRAP_MOD is None:
        _BOOTSTRAP_MOD = _load_script_module("bootstrap_dice_ci.py")
    return _BOOTSTRAP_MOD


def _wilson_mod():
    global _WILSON_MOD
    if _WILSON_MOD is None:
        _WILSON_MOD = _load_script_module("compute_wilson_ci.py")
    return _WILSON_MOD


def _resolve_round2_dir() -> Path:
    """Round-2 results dir, overridable via AGG_ROUND2_DIR (for /tmp fixtures)."""
    import os
    env = os.environ.get("AGG_ROUND2_DIR")
    if env:
        return Path(env)
    return ROUND2_DIR


def discover_kfold_summaries(round2_dir: Path | None = None) -> dict[str, Path]:
    """Map kfold run_name -> summary path for <round2>/kfold_<run_name>.json.

    run_folds.sh writes one consolidated summary per 5-fold run to
    ``results/round2/kfold_<orig-label>.json`` (orig-label = the run name from
    the 26-run matrix, e.g. ``g1_tcga_vgg16``).
    """
    base = round2_dir or ROUND2_DIR
    out: dict[str, Path] = {}
    if not base.is_dir():
        return out
    for p in sorted(base.glob("kfold_*.json")):
        run_name = p.stem[len("kfold_"):]
        out[run_name] = p
    return out


def parse_kfold_summary(path: Path) -> dict | None:
    """Parse a kfold summary into mean±SD + total val samples.

    Schema (from src/training.py::train_kfold_cv return, stored under the
    ``<dataset>_<encoder>`` run key by main.py):
      {status, dataset, encoder, k_folds, completed_folds,
       mean_val_acc, std_val_acc, mean_val_dice, std_val_dice,
       mean_val_loss, std_val_loss, fold_results: [{final_val_acc,
       final_val_dice, val_samples, ...}, ...]}
    """
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [WARN] kfold summary {path.name}: {exc}", file=sys.stderr)
        return None
    if data.get("status") not in ("completed", "running"):
        return None
    runs = data.get("runs", {})
    if not runs:
        return None
    result = None
    for _key, val in runs.items():
        if isinstance(val, dict) and "mean_val_acc" in val:
            result = val
            break
    if result is None:
        return None
    fold_results = result.get("fold_results", []) or []
    val_samples = sum(int(f.get("val_samples", 0)) for f in fold_results)
    return {
        "mean_val_acc": float(result.get("mean_val_acc", 0.0)),
        "std_val_acc": float(result.get("std_val_acc", 0.0)),
        "mean_val_dice": float(result.get("mean_val_dice", 0.0)),
        "std_val_dice": float(result.get("std_val_dice", 0.0)),
        "completed_folds": int(result.get("completed_folds", len(fold_results))),
        "val_samples": val_samples,
    }


def _read_splitter_branch(base_label: str, round2_dir: Path | None = None) -> str:
    """Read ``splitter_branch`` from the first epoch-log record of a run's dumps.

    The branch is recorded per epoch in ``<round2>/<run_label>/epoch_log.jsonl``
    (see src/training.py::train_single_run).  For kfold runs the per-fold dumps
    live under ``<base>_fold<N>of<K>/``.
    """
    base = round2_dir or ROUND2_DIR
    if not base.is_dir():
        return ""
    candidates: list[Path] = []
    base_dir = base / base_label
    if base_dir.is_dir():
        candidates.append(base_dir / "epoch_log.jsonl")
    for d in sorted(base.glob(f"{base_label}_fold*of*")):
        candidates.append(d / "epoch_log.jsonl")
    for c in candidates:
        if not c.is_file():
            continue
        try:
            with open(c) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    branch = rec.get("splitter_branch")
                    if branch:
                        return str(branch)
                    break
        except (json.JSONDecodeError, OSError):
            continue
    return ""


def _find_dump_dirs(*base_labels: str, round2_dir: Path | None = None) -> list[Path]:
    """Find per_slice_dice.jsonl dump dirs for any of the candidate base labels.

    Matches the base dir itself plus ``<base>_fold<N>of<K>`` fold dirs.
    """
    base = round2_dir or ROUND2_DIR
    if not base.is_dir():
        return []
    dirs: list[Path] = []
    for label in base_labels:
        base_dir = base / label
        if base_dir.is_dir() and (base_dir / "per_slice_dice.jsonl").is_file():
            dirs.append(base_dir)
        for d in sorted(base.glob(f"{label}_fold*of*")):
            if (d / "per_slice_dice.jsonl").is_file():
                dirs.append(d)
    return dirs


def compute_dice_ci(*base_labels: str, round2_dir: Path | None = None) -> dict:
    """Pool per-case dice across a run's dump dirs; bootstrap CI per stratum.

    Returns ``{stratum: {point, lo, hi, n}}`` for overall / positive_only /
    negative_only.  Dice CIs are strictly percentile-bootstrap (never Wilson).
    """
    bmod = _bootstrap_mod()
    dump_dirs = _find_dump_dirs(*base_labels, round2_dir=round2_dir)
    if not dump_dirs:
        return {}
    records: list[dict] = []
    for d in dump_dirs:
        records.extend(bmod.load_records(d / "per_slice_dice.jsonl"))
    if not records:
        return {}
    rows = bmod.stratum_rows(base_labels[0], records, None, 10_000, 42)
    out: dict[str, dict] = {}
    for r in rows:
        out[r["stratum"]] = {
            "point": r["point_estimate"],
            "lo": r["ci_low"],
            "hi": r["ci_high"],
            "n": r["n"],
        }
    return out


def compute_wilson_ci(acc_frac: float, n: int) -> tuple[float, float]:
    """95% Wilson score interval (Acc-only) reusing scripts/compute_wilson_ci.py."""
    wmod = _wilson_mod()
    if n <= 0:
        return (0.0, 0.0)
    successes = round(acc_frac * n)
    return wmod.wilson_score_interval(successes, n)


def augment_records_with_ci(
    records: list[dict],
    kfold_map: dict[str, dict],
    round2_dir: Path | None = None,
) -> list[dict]:
    """Attach kfold mean±SD, Wilson Acc CIs, bootstrap Dice CIs, splitter branch.

    ``kfold_map`` maps run_name -> parsed kfold summary.  Wilson CIs are
    Acc-only; Dice CIs are bootstrap-only; the two are never mixed.
    """
    for rec in records:
        run_name = rec.get("_run_name", "")
        # Defaults (empty until a source populates them).
        rec["Kfold Acc Mean (%)"] = ""
        rec["Kfold Acc SD (%)"] = ""
        rec["Kfold Dice Mean (%)"] = ""
        rec["Kfold Dice SD (%)"] = ""
        rec["Kfold N"] = ""
        rec["Acc 95% CI Lower"] = ""
        rec["Acc 95% CI Upper"] = ""
        rec["Acc 95% CI"] = ""
        rec["Dice 95% CI Lower (bootstrap)"] = ""
        rec["Dice 95% CI Upper (bootstrap)"] = ""
        rec["Dice 95% CI (bootstrap)"] = ""
        rec["Dice 95% CI Lower (pos-only)"] = ""
        rec["Dice 95% CI Upper (pos-only)"] = ""
        rec["Dice 95% CI Lower (neg-only)"] = ""
        rec["Dice 95% CI Upper (neg-only)"] = ""
        rec["Splitter Branch"] = ""

        kf = kfold_map.get(run_name)
        if kf:
            rec["Kfold Acc Mean (%)"] = round(100.0 * kf["mean_val_acc"], 2)
            rec["Kfold Acc SD (%)"] = round(100.0 * kf["std_val_acc"], 2)
            rec["Kfold Dice Mean (%)"] = round(100.0 * kf["mean_val_dice"], 2)
            rec["Kfold Dice SD (%)"] = round(100.0 * kf["std_val_dice"], 2)
            rec["Kfold N"] = kf["completed_folds"]
            # Wilson Acc CI from the kfold mean (point estimate) over total val samples.
            lo, hi = compute_wilson_ci(kf["mean_val_acc"], kf["val_samples"])
            rec["Acc 95% CI Lower"] = round(100.0 * lo, 2)
            rec["Acc 95% CI Upper"] = round(100.0 * hi, 2)
            rec["Acc 95% CI"] = f"[{100.0 * lo:.2f} - {100.0 * hi:.2f}]"
            rec["Splitter Branch"] = _read_splitter_branch(f"kfold_{run_name}", round2_dir)
        else:
            # Single-split: Wilson Acc CI from the point-estimate accuracy.
            acc_pct = rec.get("Accuracy (%)")
            n = DATASET_VAL_SIZES.get(rec.get("Dataset", ""), 0)
            if isinstance(acc_pct, (int, float)) and acc_pct != "" and n:
                lo, hi = compute_wilson_ci(float(acc_pct) / 100.0, n)
                rec["Acc 95% CI Lower"] = round(100.0 * lo, 2)
                rec["Acc 95% CI Upper"] = round(100.0 * hi, 2)
                rec["Acc 95% CI"] = f"[{100.0 * lo:.2f} - {100.0 * hi:.2f}]"
            rec["Splitter Branch"] = _read_splitter_branch(run_name, round2_dir)

        # Bootstrap Dice CIs (strictly separate from the Wilson Acc CIs above).
        dice = compute_dice_ci(f"kfold_{run_name}", run_name, round2_dir=round2_dir)
        if dice:
            ov = dice.get("overall")
            if ov:
                rec["Dice 95% CI Lower (bootstrap)"] = round(100.0 * ov["lo"], 2)
                rec["Dice 95% CI Upper (bootstrap)"] = round(100.0 * ov["hi"], 2)
                rec["Dice 95% CI (bootstrap)"] = f"[{100.0 * ov['lo']:.2f} - {100.0 * ov['hi']:.2f}]"
            pos = dice.get("positive_only")
            if pos:
                rec["Dice 95% CI Lower (pos-only)"] = round(100.0 * pos["lo"], 2)
                rec["Dice 95% CI Upper (pos-only)"] = round(100.0 * pos["hi"], 2)
            neg = dice.get("negative_only")
            if neg:
                rec["Dice 95% CI Lower (neg-only)"] = round(100.0 * neg["lo"], 2)
                rec["Dice 95% CI Upper (neg-only)"] = round(100.0 * neg["hi"], 2)
    return records


def export_csv_with_ci(records: list[dict], output_path: Path) -> None:
    """Write the CI-augmented matrix to paper/paper_results_matrix_with_ci.csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    print(f"  CSV (with CI) exported: {output_path}")


def export_csv(records: list[dict], output_path: Path) -> None:
    """Write records to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    print(f"  CSV exported: {output_path}")


def generate_latex(records: list[dict]) -> str:
    """Generate a booktabs-formatted LaTeX table string.

    The table is organized into logical sections:
      1. Main comparison (V1 vs V2 across datasets/encoders)
      2. Ablation studies (architecture, loss balancing, domain shift, alpha)
    """
    lines: list[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Experimental results across all 26 runs. "
                 "V1 uses static loss weights; V2 uses GradNorm with Macenko normalization. "
                 "Ablation rows test skip connections, loss balancing, GradNorm alpha, and color normalization.}")
    lines.append("\\label{tab:results_matrix}")
    lines.append("")
    lines.append("\\begin{tabular}{lcccccccccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Run} & \\textbf{Phase} & \\textbf{Dataset} & "
                 "\\textbf{Encoder} & \\textbf{LR} & \\textbf{GradNorm} & \\textbf{$\\alpha$} & "
                 "\\textbf{Acc \\%(\\%)} & \\textbf{Dice \\%(\\%)} & "
                 "\\textbf{Paper Acc \\%(\\%)} & \\textbf{Paper Dice \\%(\\%)} & "
                 "\\textbf{Acc $\\Delta$ \\%(\\%)} & \\textbf{Dice $\\Delta$ \\%(\\%)} \\\\")
    lines.append("\\midrule")

    main_runs: list[dict] = []
    ablation_runs: list[dict] = []

    for rec in records:
        skip = rec["Skip Connections"].lower() == "false"
        macenko = rec["Macenko"].lower() == "true"
        seg_w = rec["Seg Weight"]
        cls_w = rec["Cls Weight"]
        alpha = rec.get("GradNorm Alpha", "")

        is_ablation = (
            (skip and rec["Phase"] != "v1")
            or str(seg_w) not in ("5.0", "5", "") or str(cls_w) not in ("1.0", "1", "")
            or not macenko
            or (alpha and alpha not in ("1.0", "1", ""))
        )

        if is_ablation:
            ablation_runs.append(rec)
        else:
            main_runs.append(rec)

    phase_order = {"v1": 0, "v2": 1, "v2.1": 2}
    main_runs.sort(key=lambda r: (phase_order.get(r["Phase"], 99), r["Dataset"], r["Encoder"]))

    def ablation_key(r: dict) -> tuple:
        alpha = r.get("GradNorm Alpha", "")
        seg_w = r["Seg Weight"]
        cls_w = r["Cls Weight"]
        if r["Skip Connections"].lower() == "false" and r["Phase"] == "v2":
            return (0, r["Dataset"], r["Encoder"])
        elif str(seg_w) not in ("5.0", "5", "") or str(cls_w) not in ("1.0", "1", ""):
            return (1, str(seg_w), str(cls_w))
        elif alpha and alpha not in ("1.0", "1", ""):
            return (2, alpha)
        elif r["Macenko"].lower() == "false":
            return (3, r["Dataset"], r["Encoder"])
        else:
            return (4, r["Dataset"], r["Encoder"])

    ablation_runs.sort(key=ablation_key)

    for i, rec in enumerate(main_runs):
        gradnorm = "\\checkmark" if rec["Use GradNorm"].lower() == "true" else ""
        alpha = rec.get("GradNorm Alpha", "") or "---"
        lr = rec.get("LR", "") or "---"
        acc = f"{rec['Accuracy (%)']:.2f}" if rec["Accuracy (%)"] != "" else "---"
        dice = f"{rec['Macro Dice (%)']:.2f}" if rec["Macro Dice (%)"] != "" else "---"
        p_acc = rec["Paper Acc (%)"] if rec["Paper Acc (%)"] else "---"
        p_dice = rec["Paper Dice (%)"] if rec["Paper Dice (%)"] else "---"
        d_acc = rec["Acc Delta (%)"] if rec["Acc Delta (%)"] else "---"
        d_dice = rec["Dice Delta (%)"] if rec["Dice Delta (%)"] else "---"

        run_label = f"{rec['Dataset']}/{rec['Encoder']}"
        lines.append(f"{run_label} & {rec['Phase']} & {rec['Dataset']} & "
                     f"{rec['Encoder']} & {lr} & {gradnorm} & {alpha} & {acc} & {dice} & "
                     f"{p_acc} & {p_dice} & {d_acc} & {d_dice} \\\\")

    if main_runs and ablation_runs:
        lines.append("\\midrule")

    prev_ablation_type = ""
    for rec in ablation_runs:
        alpha = rec.get("GradNorm Alpha", "")
        seg_w = rec["Seg Weight"]
        cls_w = rec["Cls Weight"]
        if rec["Skip Connections"].lower() == "false" and rec["Phase"] == "v2":
            ablation_type = "arch"
        elif str(seg_w) not in ("5.0", "5", "") or str(cls_w) not in ("1.0", "1", ""):
            ablation_type = "loss"
        elif alpha and alpha not in ("1.0", "1", ""):
            ablation_type = "alpha"
        elif rec["Macenko"].lower() == "false":
            ablation_type = "domain"
        else:
            ablation_type = "other"

        if ablation_type != prev_ablation_type and prev_ablation_type != "":
            lines.append("\\cmidrule(lr){1-13}")
            prev_ablation_type = ablation_type

        gradnorm = "\\checkmark" if rec["Use GradNorm"].lower() == "true" else ""
        alpha_str = alpha if alpha else "---"
        lr = rec.get("LR", "") or "---"
        acc = f"{rec['Accuracy (%)']:.2f}" if rec["Accuracy (%)"] != "" else "---"
        dice = f"{rec['Macro Dice (%)']:.2f}" if rec["Macro Dice (%)"] != "" else "---"
        p_acc = rec["Paper Acc (%)"] if rec["Paper Acc (%)"] else "---"
        p_dice = rec["Paper Dice (%)"] if rec["Paper Dice (%)"] else "---"
        d_acc = rec["Acc Delta (%)"] if rec["Acc Delta (%)"] else "---"
        d_dice = rec["Dice Delta (%)"] if rec["Dice Delta (%)"] else "---"

        if ablation_type == "arch":
            run_label = f"{rec['Dataset']}/{rec['Encoder']} $\\times$"
        elif ablation_type == "loss":
            ratio = rec.get("Lambda Ratio (Seg:Cls)", "")
            run_label = f"{rec['Dataset']}/{rec['Encoder']} (seg:cls={ratio})"
        elif ablation_type == "alpha":
            run_label = f"{rec['Dataset']}/{rec['Encoder']} ($\\alpha$={alpha})"
        elif ablation_type == "domain":
            run_label = f"{rec['Dataset']}/{rec['Encoder']} $\\diamond$"
        else:
            run_label = f"{rec['Dataset']}/{rec['Encoder']}"

        lines.append(f"{run_label} & {rec['Phase']} & {rec['Dataset']} & "
                     f"{rec['Encoder']} & {lr} & {gradnorm} & {alpha_str} & {acc} & {dice} & "
                     f"{p_acc} & {p_dice} & {d_acc} & {d_dice} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")
    lines.append("\\small{Notes: $\\times$ = skip connections ablated; "
                 "$\\diamond$ = Macenko normalization disabled; "
                 "$\\alpha$ = GradNorm weighting parameter. "
                 "Acc $\\Delta$ and Dice $\\Delta$ are differences from paper reference targets.}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def export_latex(latex_str: str, output_path: Path) -> None:
    """Write LaTeX table to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex_str)
    print(f"  LaTeX table exported: {output_path}")


def main() -> int:
    """Entry point."""
    print("=" * 60)
    print("  Results Aggregator - 26-Run Validation")
    print("=" * 60)

    # Phase 1: Validate all 26 expected runs
    print(f"\n{'=' * 60}")
    print(f"  PHASE 1: Validating all {len(EXPECTED_RUNS)} expected runs")
    print(f"{'=' * 60}")

    all_valid, issues, lr_map, valid_count, failed_count = validate_all_runs()

    if issues:
        print(f"\n  Validation issues found:")
        for issue in issues:
            print(issue)

    print(f"\n  Validation summary: {valid_count}/{len(EXPECTED_RUNS)} runs valid, {failed_count} failed")

    if not all_valid:
        print(f"\n[ERROR] Not all {len(EXPECTED_RUNS)} runs are valid.")
        print(f"  Cannot aggregate results until all runs complete successfully.")
        print(f"  Missing/failed runs must be re-run before aggregation.")
        print()
        return 1

    # Phase 2: Discover and parse summary files
    print(f"\n{'=' * 60}")
    print(f"  PHASE 2: Parsing summary files")
    print(f"{'=' * 60}")

    summary_files = discover_summary_files()
    if not summary_files:
        print(f"\n[ERROR] No summary files found in {CHECKPOINT_DIR}/", file=sys.stderr)
        return 1

    print(f"\nDiscovered {len(summary_files)} summary file(s):")
    for sf in summary_files:
        lr = lr_map.get(sf.name, "")
        lr_tag = f" [LR={lr}]" if lr else ""
        print(f"  - {sf.name}{lr_tag}")

    all_records: list[dict] = []
    for sf in summary_files:
        print(f"\nParsing {sf.name} ...")
        lr = lr_map.get(sf.name, "")
        records = parse_summary_file(sf, lr=lr)
        print(f"  Extracted {len(records)} run(s)")
        all_records.extend(records)

    if not all_records:
        print("\n[ERROR] No run records extracted.", file=sys.stderr)
        return 1

    # Phase 3: Deduplicate
    print(f"\n{'=' * 60}")
    print(f"  PHASE 3: Deduplication")
    print(f"{'=' * 60}")

    print(f"\nTotal records before dedup: {len(all_records)}")
    unique_records = deduplicate_records(all_records)
    print(f"Unique records after dedup: {len(unique_records)}")

    if len(unique_records) < len(EXPECTED_RUNS):
        print(f"\n  [WARN] Only {len(unique_records)} unique records after dedup "
              f"(expected {len(EXPECTED_RUNS)}). Some runs may share identical configs.")

    # Phase 4: Export
    print(f"\n{'=' * 60}")
    print(f"  PHASE 4: Exporting results")
    print(f"{'=' * 60}")

    print(f"\nExporting CSV ...")
    export_csv(unique_records, CSV_OUTPUT)

    print(f"Generating LaTeX table ...")
    latex_str = generate_latex(unique_records)
    export_latex(latex_str, LATEX_OUTPUT)

    # Phase 5: Round-2 CI augmentation (kfold mean±SD + Wilson Acc + bootstrap Dice).
    print(f"\n{'=' * 60}")
    print(f"  PHASE 5: CI augmentation (kfold CV + Wilson Acc + bootstrap Dice)")
    print(f"{'=' * 60}")

    round2_dir = _resolve_round2_dir()
    kfold_paths = discover_kfold_summaries(round2_dir)
    kfold_map: dict[str, dict] = {}
    for run_name, p in kfold_paths.items():
        parsed = parse_kfold_summary(p)
        if parsed:
            kfold_map[run_name] = parsed
            print(f"  [OK]   kfold {run_name}: {parsed['completed_folds']} folds, "
                  f"acc={100.0 * parsed['mean_val_acc']:.2f}±{100.0 * parsed['std_val_acc']:.2f}")
        else:
            print(f"  [WARN] kfold {run_name}: summary not usable ({p.name})")
    if not kfold_map:
        print("  [INFO] No kfold summaries found; Wilson Acc CIs from single-split point estimates.")

    augmented = augment_records_with_ci(unique_records, kfold_map, round2_dir=round2_dir)
    export_csv_with_ci(augmented, CSV_WITH_CI_OUTPUT)

    print("\n" + "=" * 60)
    print("  Aggregation complete.")
    print(f"  CSV:   {CSV_OUTPUT}")
    print(f"  CSV+CI:{CSV_WITH_CI_OUTPUT}")
    print(f"  LaTeX: {LATEX_OUTPUT}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())