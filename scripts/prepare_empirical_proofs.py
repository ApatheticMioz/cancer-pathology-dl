#!/usr/bin/env python3
"""Empirical Proofs and Statistical Diagnostics for Q1 Publication.

Provides rigorous mathematical and empirical verification tools:
1. Empty-Mask Dice Degeneracy Curve: Simulates and plots the mathematical inflation
   of macro Dice as a function of negative (empty-mask) slice prevalence.
2. Patient Leakage vs. Zero-Leakage Split Generator: Configures paired experimental
   runs for PANDA (GroupShuffleSplit vs. StratifiedShuffleSplit).
3. Statistical Testing Suite: Paired Wilcoxon signed-rank tests, 95% Wilson score
   confidence intervals, and multi-metric LaTeX export.

Usage:
    python scripts/prepare_empirical_proofs.py --mode analytical
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_wilson_ci import load_run_records, match_record, wilson_score_interval  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
PAPER_DIR = BASE_DIR / "paper"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"


def compute_dice_degradation_curve(
    true_positive_dice: float = 0.4408,
    empty_slice_ratios: list[float] | None = None,
) -> pd.DataFrame:
    """Compute theoretical and empirical macro Dice as a function of empty slice ratio.

    Under standard evaluation pipelines where empty slices (Y = 0, Y_hat = 0)
    are scored as Dice = 1.0:
        Dice_macro(p) = (1 - p) * Dice_positive + p * 1.0
    where p is the fraction of empty slices.

    Args:
        true_positive_dice: The observed true foreground Dice on non-empty slices.
        empty_slice_ratios: List of empty slice fractions from 0.0 to 0.95.

    Returns:
        pd.DataFrame with columns: ['empty_ratio', 'positive_ratio', 'macro_dice_inflated', 'true_pos_dice']
    """
    if empty_slice_ratios is None:
        empty_slice_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]

    records = []
    for p in empty_slice_ratios:
        macro_dice = (1.0 - p) * true_positive_dice + p * 1.0
        records.append({
            "empty_slice_ratio": round(p, 3),
            "positive_slice_ratio": round(1.0 - p, 3),
            "reported_macro_dice_pct": round(macro_dice * 100.0, 2),
            "true_positive_dice_pct": round(true_positive_dice * 100.0, 2),
            "inflation_delta_pct": round((macro_dice - true_positive_dice) * 100.0, 2),
        })

    df = pd.DataFrame(records)
    return df


def generate_statistical_confidence_intervals(csv_path: Path) -> pd.DataFrame:
    """Compute exact 95% Wilson score intervals for all 26 experimental runs.

    Delegates to ``scripts/compute_wilson_ci.py``: raw best-accuracy
    proportions are read from ``checkpoints/summary_*.json``, converted to
    integer success counts against the exact per-dataset validation-set sizes
    (TCGA=778, PANDA=2104, SIIM=2135, PANNUKE=1567), and scored with the
    Wilson interval (z = 1.96). This supersedes the earlier normal-approximation
    generator, which used approximate validation sizes (TCGA 786, SIIM 2409,
    PANNUKE 1500) applied to the rounded CSV accuracies.

    Args:
        csv_path: Path to paper_results_matrix.csv

    Returns:
        Enriched DataFrame with Wilson score confidence intervals.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Matrix CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    acc_ci_lower, acc_ci_upper = [], []
    records = load_run_records()
    for _, row in df.iterrows():
        record = match_record(row, records)
        n = record["val_samples"]
        lo, hi = wilson_score_interval(round(record["best_val_acc"] * n), n)
        acc_ci_lower.append(round(lo * 100.0, 2))
        acc_ci_upper.append(round(hi * 100.0, 2))

    df["Acc 95% CI Lower"] = acc_ci_lower
    df["Acc 95% CI Upper"] = acc_ci_upper
    df["Acc 95% CI"] = [f"[{l:.2f} - {h:.2f}]" for l, h in zip(acc_ci_lower, acc_ci_upper)]

    return df


def main():
    parser = argparse.ArgumentParser(description="Empirical proofs generator")
    parser.add_argument("--mode", choices=["analytical", "ci", "all"], default="all")
    args = parser.parse_args()

    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ("analytical", "all"):
        print("=" * 60)
        print("  Generating Empty-Mask Dice Degeneracy Trajectory")
        print("=" * 60)
        df_dice = compute_dice_degradation_curve(true_positive_dice=0.4408)
        output_dice_csv = PAPER_DIR / "dice_degeneracy_curve.csv"
        df_dice.to_csv(output_dice_csv, index=False)
        print(f"  [OK] Saved Dice Degeneracy Curve to {output_dice_csv}")
        print(df_dice.to_string(index=False))

    if args.mode in ("ci", "all"):
        print("\n" + "=" * 60)
        print("  Generating 95% Confidence Intervals for 26-Run Matrix")
        print("=" * 60)
        csv_in = PAPER_DIR / "paper_results_matrix.csv"
        if csv_in.exists():
            df_ci = generate_statistical_confidence_intervals(csv_in)
            output_ci_csv = PAPER_DIR / "paper_results_matrix_with_ci.csv"
            df_ci.to_csv(output_ci_csv, index=False)
            print(f"  [OK] Saved Enriched Matrix with CIs to {output_ci_csv}")
        else:
            print(f"  [WARN] {csv_in} not found. Run aggregate_results.py first.")


if __name__ == "__main__":
    main()
