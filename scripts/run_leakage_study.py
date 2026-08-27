#!/usr/bin/env python3
"""Controlled Data Leakage Replication Study for PANDA Prostate Histology.

Compares two split regimes under identical model architecture and hyperparameters:
1. 'isolated' (Clean): GroupShuffleSplit on whole-slide image_id (zero patient leakage).
2. 'leaked' (Reproducing Rhanoui et al.): StratifiedShuffleSplit on raw patches without
   patient/slide grouping, allowing identical biopsy tissue textures in train and val.

Usage:
    python scripts/run_leakage_study.py --split-mode isolated --dry-run
    python scripts/run_leakage_study.py --split-mode leaked --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import DATASET_META
from src.data import load_dataset_bundle, make_group_split
from src.models import MultiTaskUNet


def parse_args():
    parser = argparse.ArgumentParser(description="PANDA Data Leakage Verification Study")
    parser.add_argument(
        "--split-mode",
        choices=["isolated", "leaked"],
        default="isolated",
        help="Split strategy: 'isolated' (GroupKFold on slide_id) or 'leaked' (Random patch split)",
    )
    parser.add_argument("--encoder", choices=["vgg16", "mobilenet_v2"], default="vgg16")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Print configuration and exit without training")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("  PANDA Controlled Data Leakage Replication Study")
    print(f"  Split Mode: {args.split_mode.upper()}")
    print(f"  Encoder:    {args.encoder}")
    print(f"  LR:         {args.lr}")
    print("=" * 65)

    panda_root = BASE_DIR / "data" / "PANDA"
    bundle = load_dataset_bundle("panda", panda_root, skip_macenko=True)
    labels = bundle["labels"]
    groups = bundle["groups"]

    if args.split_mode == "isolated":
        tr_idx, val_idx = make_group_split(labels, groups, seed=args.seed, test_size=0.2)
        overlap = set(groups[tr_idx]) & set(groups[val_idx])
        print(f"  [ISOLATED] Train samples: {len(tr_idx)}, Val samples: {len(val_idx)}")
        print(f"  [ISOLATED] Unique train slides: {len(set(groups[tr_idx]))}, Val slides: {len(set(groups[val_idx]))}")
        print(f"  [ISOLATED] Inter-split patient overlap: {len(overlap)} (Strict 0.0% guarantee)")
    else:
        # Leaked split: bypass group_id by assigning each patch a unique index
        dummy_groups = np.arange(len(labels))
        tr_idx, val_idx = make_group_split(labels, dummy_groups, seed=args.seed, test_size=0.2)
        # Compute true patient overlap
        tr_slides = set(groups[tr_idx])
        val_slides = set(groups[val_idx])
        overlap_slides = tr_slides & val_slides
        print(f"  [LEAKED] Train samples: {len(tr_idx)}, Val samples: {len(val_idx)}")
        print(f"  [LEAKED] Overlapping slide cores present in BOTH train and val: {len(overlap_slides)}")
        print(f"  [LEAKED] Percentage of val slides contaminated: {len(overlap_slides) / len(val_slides) * 100:.2f}%")

    if args.dry_run:
        print("\n  [INFO] Dry run complete. Split verified.")
        return 0

    print("\n  Ready for training execution with standard PyTorch loop.")
    return 0


if __name__ == "__main__":
    import numpy as np
    sys.exit(main())
