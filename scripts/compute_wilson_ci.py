"""Recompute 95% Wilson score intervals for the 26-run results matrix.

Supersedes the normal-approximation intervals emitted by
``scripts/prepare_empirical_proofs.py::generate_statistical_confidence_intervals``,
which (a) used approximate validation-set sizes (TCGA 786 vs. actual 778,
SIIM 2409 vs. actual 2135, PANNUKE 1500 vs. actual 1567) and (b) consumed the
already-rounded CSV accuracies, while the manuscript described all intervals
as "95% Wilson confidence intervals".

This script reads the raw ``best_val_acc`` proportions from
``checkpoints/summary_*.json``, converts them to integer success counts against
the exact per-dataset validation-set sizes (pinned from the run logs:
TCGA=778, PANDA=2104, SIIM=2135, PANNUKE=1567), and applies the same Wilson
score interval used by ``scripts/run_canonical_gradnorm_panda.py``
(z = 1.96, two-sided). With this scheme the regenerated interval for Run 18
reproduces the previously published [27.14, 31.02] exactly.

Usage:
    python3 scripts/compute_wilson_ci.py [--write]

Without ``--write`` the script only prints the old -> new diff and the
overlap/disjointness verdicts for the run pairs quoted in the manuscript.
With ``--write`` it rewrites ``paper/paper_results_matrix_with_ci.csv`` in
place, replacing only the three trailing CI columns.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATRIX_CSV = PROJECT_ROOT / "paper" / "paper_results_matrix_with_ci.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Exact validation-set sizes pinned from logs/run_* training lines
# (e.g. "SIIM x vgg16 | samples=10675 train=8540 val=2135 bs=32").
DATASET_VAL_SIZES = {
    "TCGA": 778,
    "PANDA": 2104,
    "SIIM": 2135,
    "PANNUKE": 1567,
}

Z = 1.96

# Run pairs whose interval relationship is asserted in the manuscript prose.
PROSE_PAIRS = [
    ("skip TCGA (08 vs 25)", 8, 25),
    ("skip PANDA (10 vs 26)", 10, 26),
    ("Macenko PANDA (10 vs 23)", 10, 23),
    ("Macenko PanNuke (16 vs 24)", 16, 24),
    ("LR isolation (17 vs 20)", 17, 20),
    ("GradNorm collapse (18 vs 20)", 18, 20),
]


def wilson_score_interval(successes: int, total: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval; identical to run_canonical_gradnorm_panda.py."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z ** 2) / total
    centre = (p + (z ** 2) / (2.0 * total)) / denom
    spread = (z * ((p * (1.0 - p) / total + (z ** 2) / (4.0 * total ** 2)) ** 0.5)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def load_run_records() -> list[dict]:
    """Flatten all per-run records from checkpoints/summary_*.json."""
    records = []
    for path in sorted(CHECKPOINT_DIR.glob("summary_*.json")):
        with open(path) as fh:
            summary = json.load(fh)
        args_ = summary.get("args", {})
        for key, run in summary.get("runs", {}).items():
            records.append(
                {
                    "source": path.name,
                    "dataset": str(run.get("dataset", "")).lower(),
                    "encoder": str(run.get("encoder", "")).strip().lower(),
                    "best_val_acc": float(run["best_val_acc"]),
                    "val_samples": int(run["val_samples"]),
                    "use_gradnorm": bool(args_.get("use_gradnorm", False)),
                    "lr": float(args_.get("lr", -1)),
                    "no_macenko": bool(args_.get("no_macenko", False)),
                    "no_skip": bool(run.get("skip_connections_ablated", False)),
                }
            )
    return records


def match_record(row: dict, records: list[dict]) -> dict:
    """Match a CSV row to its run record by dataset/encoder/config and rounded accuracy."""
    dataset = str(row["Dataset"]).lower()
    encoder = str(row["Encoder"]).strip().lower()
    encoder = "mobilenet_v2" if encoder in {"mobilenetv2", "mobilenet_v2"} else encoder
    acc_pct = float(row["Accuracy (%)"])
    use_gn = str(row["Use GradNorm"]).strip().lower() == "true"
    lr = float(row["LR"])
    # CSV columns state whether Macenko/skips were USED; artifact fields state
    # whether they were disabled (no_macenko / skip_connections_ablated).
    macenko_used = str(row["Macenko"]).strip().lower() == "true"
    skip_used = str(row["Skip Connections"]).strip().lower() == "true"

    candidates = [
        r
        for r in records
        if r["dataset"] == dataset
        and r["encoder"] == encoder
        and abs(r["best_val_acc"] * 100.0 - acc_pct) < 0.005
        and r["use_gradnorm"] == use_gn
        and abs(r["lr"] - lr) < 1e-12
        and r["no_macenko"] == (not macenko_used)
        and r["no_skip"] == (not skip_used)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Row match failed (dataset={dataset}, encoder={encoder}, acc={acc_pct}): "
            f"{len(candidates)} candidates"
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite the CI CSV in place")
    args = parser.parse_args()

    records = load_run_records()
    with open(MATRIX_CSV, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    lows_new, highs_new, lows_old, highs_old = [], [], [], []
    for idx, row in enumerate(rows, start=1):
        record = match_record(row, records)
        n = record["val_samples"]
        expected_n = DATASET_VAL_SIZES[str(row["Dataset"]).upper()]
        if n != expected_n:
            print(f"WARNING run {idx}: artifact val_samples={n} != pinned {expected_n}")
        acc = record["best_val_acc"]
        successes = round(acc * n)
        lo, hi = wilson_score_interval(successes, n)
        lows_new.append(round(lo * 100.0, 2))
        highs_new.append(round(hi * 100.0, 2))
        lows_old.append(float(row["Acc 95% CI Lower"]))
        highs_old.append(float(row["Acc 95% CI Upper"]))
        row["_run"] = idx
        row["_k"] = successes
        row["_n"] = n

    print(f"{'run':>3}  {'dataset':8} {'n':>5} {'k':>5}  {'old CI':18} {'new CI (Wilson)':18} {'changed'}")
    changed = 0
    for row, lo_o, hi_o, lo_n, hi_n in zip(rows, lows_old, highs_old, lows_new, highs_new):
        is_changed = (lo_o, hi_o) != (lo_n, hi_n)
        changed += int(is_changed)
        print(
            f"{row['_run']:>3}  {str(row['Dataset']):8} {row['_n']:>5} {row['_k']:>5}  "
            f"[{lo_o:5.2f}, {hi_o:5.2f}]   [{lo_n:5.2f}, {hi_n:5.2f}]   "
            f"{'YES' if is_changed else 'no'}"
        )
    print(f"\n{changed}/26 intervals changed")

    print("\nProse-pair overlap verdicts (Wilson):")
    for label, a, b in PROSE_PAIRS:
        ra, rb = rows[a - 1], rows[b - 1]
        ia = (lows_new[a - 1], highs_new[a - 1])
        ib = (lows_new[b - 1], highs_new[b - 1])
        overlaps = not (ia[1] < ib[0] or ib[1] < ia[0])
        print(f"  {label:28} {ia} vs {ib} -> {'OVERLAP' if overlaps else 'DISJOINT'}")

    if args.write:
        for row, lo_n, hi_n in zip(rows, lows_new, highs_new):
            row["Acc 95% CI Lower"] = f"{lo_n:.2f}"
            row["Acc 95% CI Upper"] = f"{hi_n:.2f}"
            row["Acc 95% CI"] = f"[{lo_n:.2f} - {hi_n:.2f}]"
        for row in rows:
            for key in ("_run", "_k", "_n"):
                row.pop(key, None)
        with open(MATRIX_CSV, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {MATRIX_CSV}")


if __name__ == "__main__":
    main()
