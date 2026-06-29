#!/usr/bin/env python3
"""
recover_forensics.py
Data forensics script to recover experimental run metrics from epoch_log.jsonl.

Parses checkpoints_v2/epoch_log.jsonl, filters for June 27 non-smoke-test runs,
groups by epoch resets, and maps to the expected 20-run chronological sequence
from run_all_experiments.sh.

Outputs:
  - paper/recovered_results_matrix.csv
  - paper/recovered_latex_table.txt
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_PATH = Path(__file__).resolve().parent.parent / "checkpoints_v2" / "epoch_log.jsonl"
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "paper" / "recovered_results_matrix.csv"
OUTPUT_LATEX = Path(__file__).resolve().parent.parent / "paper" / "recovered_latex_table.txt"

# Target date filter
TARGET_DATE_PREFIX = "2026-06-27"

# The exact 20-run chronological sequence from run_all_experiments.sh
EXPECTED_RUNS = [
    # Phase 1: Leak-Free Baselines V1
    {"run_id": 1,  "phase": "Phase 1", "sub_phase": "1",    "dataset": "tcga",       "encoder": "vgg16",       "description": "Baseline",              "extra": ""},
    {"run_id": 2,  "phase": "Phase 1", "sub_phase": "1",    "dataset": "tcga",       "encoder": "mobilenet_v2", "description": "Baseline",              "extra": ""},
    {"run_id": 3,  "phase": "Phase 1", "sub_phase": "1",    "dataset": "panda",      "encoder": "vgg16",       "description": "Baseline",              "extra": ""},
    {"run_id": 4,  "phase": "Phase 1", "sub_phase": "1",    "dataset": "panda",      "encoder": "mobilenet_v2", "description": "Baseline",              "extra": ""},
    {"run_id": 5,  "phase": "Phase 1", "sub_phase": "1",    "dataset": "siim",       "encoder": "vgg16",       "description": "Baseline",              "extra": ""},
    {"run_id": 6,  "phase": "Phase 1", "sub_phase": "1",    "dataset": "siim",       "encoder": "mobilenet_v2", "description": "Baseline",              "extra": ""},
    # Phase 2: Enhanced System V2 (GradNorm + Macenko)
    {"run_id": 7,  "phase": "Phase 2", "sub_phase": "2",    "dataset": "tcga",       "encoder": "vgg16",       "description": "GradNorm + Macenko",    "extra": ""},
    {"run_id": 8,  "phase": "Phase 2", "sub_phase": "2",    "dataset": "tcga",       "encoder": "mobilenet_v2", "description": "GradNorm + Macenko",    "extra": ""},
    {"run_id": 9,  "phase": "Phase 2", "sub_phase": "2",    "dataset": "panda",      "encoder": "vgg16",       "description": "GradNorm + Macenko",    "extra": ""},
    {"run_id": 10, "phase": "Phase 2", "sub_phase": "2",    "dataset": "panda",      "encoder": "mobilenet_v2", "description": "GradNorm + Macenko",    "extra": ""},
    {"run_id": 11, "phase": "Phase 2", "sub_phase": "2",    "dataset": "siim",       "encoder": "vgg16",       "description": "GradNorm + Macenko",    "extra": ""},
    {"run_id": 12, "phase": "Phase 2", "sub_phase": "2",    "dataset": "siim",       "encoder": "mobilenet_v2", "description": "GradNorm + Macenko",    "extra": ""},
    # Phase 3: Control V2.1 (PanNuke baseline)
    {"run_id": 13, "phase": "Phase 3", "sub_phase": "3",    "dataset": "pannuke",    "encoder": "vgg16",       "description": "V2.1 Control",          "extra": ""},
    {"run_id": 14, "phase": "Phase 3", "sub_phase": "3",    "dataset": "pannuke",    "encoder": "mobilenet_v2", "description": "V2.1 Control",          "extra": ""},
    # Phase 4A: Architecture Dependency Ablation
    {"run_id": 15, "phase": "Phase 4A", "sub_phase": "4A",  "dataset": "tcga",       "encoder": "mobilenet_v2", "description": "No Skip Connections",   "extra": "--no-skip-connections"},
    # Phase 4B: Loss Balancing / Gradient Starvation
    {"run_id": 16, "phase": "Phase 4B", "sub_phase": "4B",  "dataset": "panda",      "encoder": "vgg16",       "description": "Static Loss (Seg=1, Cls=1)",  "extra": "--lambda-seg 1 --lambda-cls 1"},
    {"run_id": 17, "phase": "Phase 4B", "sub_phase": "4B",  "dataset": "panda",      "encoder": "vgg16",       "description": "Static Loss (Seg=5, Cls=1)",  "extra": "--lambda-seg 5 --lambda-cls 1"},
    {"run_id": 18, "phase": "Phase 4B", "sub_phase": "4B",  "dataset": "panda",      "encoder": "vgg16",       "description": "Static Loss (Seg=1, Cls=5)",  "extra": "--lambda-seg 1 --lambda-cls 5"},
    # Phase 4C: Domain Shift Ablation
    {"run_id": 19, "phase": "Phase 4C", "sub_phase": "4C",  "dataset": "panda",      "encoder": "mobilenet_v2", "description": "No Macenko",            "extra": "--no-macenko"},
    {"run_id": 20, "phase": "Phase 4C", "sub_phase": "4C",  "dataset": "pannuke",    "encoder": "mobilenet_v2", "description": "No Macenko",            "extra": "--no-macenko"},
]


def parse_log():
    """Parse epoch_log.jsonl and return filtered lines."""
    lines = []
    with open(LOG_PATH, "r") as f:
        for line_num, raw_line in enumerate(f, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            # Skip smoke tests
            if entry.get("smoke_test", False):
                continue

            # Skip non-June-27 entries
            ts = entry.get("timestamp", "")
            if not ts.startswith(TARGET_DATE_PREFIX):
                continue

            entry["_line"] = line_num
            lines.append(entry)

    return lines


def group_into_runs(lines):
    """
    Group log entries into distinct runs.
    A new run begins whenever epoch resets to 1.
    Returns list of runs, each run is a list of epoch entries.
    """
    runs = []
    current_run = []

    for entry in lines:
        if entry["epoch"] == 1 and current_run:
            # Epoch reset -> new run
            runs.append(current_run)
            current_run = []
        current_run.append(entry)

    if current_run:
        runs.append(current_run)

    return runs


def extract_run_signature(run_entries):
    """Extract dataset and encoder from the first epoch of a run."""
    first = run_entries[0]
    return {
        "dataset": first["dataset"],
        "encoder": first["encoder"],
        "start_timestamp": first["timestamp"],
        "total_epochs": len(run_entries),
    }


def extract_best_metrics(run_entries):
    """Extract the best_vl_acc and best_vl_dice from the final epoch of a run."""
    last = run_entries[-1]
    return {
        "best_vl_acc": last.get("best_vl_acc"),
        "best_vl_dice": last.get("best_vl_dice"),
        "best_vl_loss": last.get("best_vl_loss"),
        "final_epoch": last["epoch"],
        "final_timestamp": last["timestamp"],
    }


def match_run_to_expected(run_entries, expected):
    """
    Check if a parsed run matches an expected run definition.
    Matches on dataset and encoder.
    """
    sig = extract_run_signature(run_entries)
    return (sig["dataset"] == expected["dataset"] and
            sig["encoder"] == expected["encoder"])


def map_runs_to_sequence(parsed_runs):
    """
    Map parsed runs to the expected 20-run sequence by strict chronological order.
    Uses dataset+encoder signature matching in sequence order.
    """
    # Build the mapped result
    mapped = []
    used_indices = set()

    for exp in EXPECTED_RUNS:
        best_match_idx = None
        best_match_score = -1

        for i, run in enumerate(parsed_runs):
            if i in used_indices:
                continue
            if match_run_to_expected(run, exp):
                # Match found - since we process in order, first match wins
                best_match_idx = i
                break

        if best_match_idx is not None:
            used_indices.add(best_match_idx)
            sig = extract_run_signature(parsed_runs[best_match_idx])
            metrics = extract_best_metrics(parsed_runs[best_match_idx])

            mapped.append({
                "run_id": exp["run_id"],
                "phase": exp["phase"],
                "sub_phase": exp["sub_phase"],
                "dataset": exp["dataset"],
                "encoder": exp["encoder"],
                "description": exp["description"],
                "extra_flags": exp["extra"],
                "status": "FOUND",
                "start_timestamp": sig["start_timestamp"],
                "end_timestamp": metrics["final_timestamp"],
                "total_epochs": sig["total_epochs"],
                "best_vl_acc": metrics["best_vl_acc"],
                "best_vl_dice": metrics["best_vl_dice"],
                "best_vl_loss": metrics["best_vl_loss"],
            })
        else:
            mapped.append({
                "run_id": exp["run_id"],
                "phase": exp["phase"],
                "sub_phase": exp["sub_phase"],
                "dataset": exp["dataset"],
                "encoder": exp["encoder"],
                "description": exp["description"],
                "extra_flags": exp["extra"],
                "status": "NOT_FOUND",
                "start_timestamp": None,
                "end_timestamp": None,
                "total_epochs": None,
                "best_vl_acc": None,
                "best_vl_dice": None,
                "best_vl_loss": None,
            })

    return mapped


def write_csv(mapped, path):
    """Write the mapped results to a CSV file."""
    fieldnames = [
        "run_id", "phase", "sub_phase", "dataset", "encoder", "description",
        "extra_flags", "status", "start_timestamp", "end_timestamp",
        "total_epochs", "best_vl_acc", "best_vl_dice", "best_vl_loss"
    ]

    os.makedirs(path.parent, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in mapped:
            writer.writerow(row)

    print(f"CSV written to: {path}")


def write_latex_table(mapped, path):
    """Write a polished LaTeX table with booktabs formatting."""
    os.makedirs(path.parent, exist_ok=True)

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Complete Experimental Results Matrix -- 20-Run Forensic Recovery}")
    lines.append("\\label{tab:results_matrix}")
    lines.append("\\begin{tabular}{lllllllll}")
    lines.append("\\toprule")
    lines.append("\\textbf{Run} & \\textbf{Phase} & \\textbf{Dataset} & \\textbf{Encoder} & \\textbf{Configuration} & \\textbf{Status} & \\textbf{Epochs} & \\textbf{Best VL Acc} & \\textbf{Best VL Dice} \\\\")
    lines.append("\\midrule")

    current_phase = None
    for row in mapped:
        # Phase separator
        if row["phase"] != current_phase:
            if current_phase is not None:
                lines.append("\\midrule")
            current_phase = row["phase"]

        run_id = str(row["run_id"])
        phase = row["phase"]
        dataset = row["dataset"].upper()
        encoder = row["encoder"].replace("_", "\\_")
        config = row["description"]

        if row["status"] == "FOUND":
            status = "\\checkmark"
            epochs = str(row["total_epochs"])
            acc = f"{row['best_vl_acc']:.4f}" if row["best_vl_acc"] is not None else "N/A"
            dice = f"{row['best_vl_dice']:.4f}" if row["best_vl_dice"] is not None else "N/A"
        else:
            status = "\\textbf{NOT FOUND}"
            epochs = "---"
            acc = "---"
            dice = "---"

        lines.append(f"{run_id} & {phase} & {dataset} & {encoder} & {config} & {status} & {epochs} & {acc} & {dice} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    latex_content = "\n".join(lines)

    with open(path, "w") as f:
        f.write(latex_content)

    print(f"LaTeX table written to: {path}")
    return latex_content


def main():
    print("=" * 70)
    print("DATA FORENSICS: Recovering Experimental Run Matrix")
    print("=" * 70)

    # Step 1: Parse and filter
    print(f"\n[1] Parsing {LOG_PATH} ...")
    lines = parse_log()
    print(f"    Found {len(lines)} non-smoke-test entries from {TARGET_DATE_PREFIX}")

    # Step 2: Group into runs
    print("\n[2] Grouping into distinct runs (by epoch reset to 1) ...")
    runs = group_into_runs(lines)
    print(f"    Identified {len(runs)} distinct runs")

    # Print run signatures for verification
    print("\n    Run signatures (chronological order):")
    for i, run in enumerate(runs):
        sig = extract_run_signature(run)
        metrics = extract_best_metrics(run)
        print(f"      Run #{i+1}: {sig['dataset']:8s} | {sig['encoder']:14s} | "
              f"epochs={sig['total_epochs']:2d} | "
              f"best_vl_acc={metrics['best_vl_acc']} | "
              f"best_vl_dice={metrics['best_vl_dice']} | "
              f"start={sig['start_timestamp']}")

    # Step 3: Map to expected sequence
    print("\n[3] Mapping to expected 20-run sequence ...")
    mapped = map_runs_to_sequence(runs)

    found_count = sum(1 for m in mapped if m["status"] == "FOUND")
    missing_count = sum(1 for m in mapped if m["status"] == "NOT_FOUND")
    print(f"    Matched: {found_count}/20 runs")
    print(f"    Missing: {missing_count}/20 runs")

    # Print missing runs
    missing = [m for m in mapped if m["status"] == "NOT_FOUND"]
    if missing:
        print("\n    Missing runs:")
        for m in missing:
            print(f"      Run #{m['run_id']}: {m['phase']} | {m['dataset']} | {m['encoder']} | {m['description']}")

    # Step 4: Write outputs
    print("\n[4] Writing output files ...")
    write_csv(mapped, OUTPUT_CSV)
    latex_content = write_latex_table(mapped, OUTPUT_LATEX)

    # Step 5: Print LaTeX to console
    print("\n" + "=" * 70)
    print("LATEX TABLE OUTPUT")
    print("=" * 70)
    print(latex_content)

    print("\n" + "=" * 70)
    print("FORENSIC RECOVERY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()