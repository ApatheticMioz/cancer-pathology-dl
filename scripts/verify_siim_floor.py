"""Verify the SIIM-ACR empty-mask floor mechanism analytically and empirically.

The four SIIM benchmark runs (05, 06, 11, 12) all report the byte-identical
``best_val_dice = 0.7774375410222295``. The claimed mechanism is total
negative-class collapse: the segmentation head predicts all-empty masks, so
under the ``dice_coefficient`` convention (``src/metrics.py``: union == 0 ->
score 1.0) each empty-ground-truth slice scores 1.0 and each positive slice
scores 0.0, and the reported batch-mean average equals a weighted empty-slice
fraction determined by the deterministic validation ordering (shuffle=False,
batch size 32: 66 batches of 32 + 1 batch of 23).

This script replicates the exact pipeline -- ``parse_siim`` filtering, the
``make_group_split`` fallback to ``StratifiedShuffleSplit`` when every sample
is its own group (seed 42, test_size 0.2), validation-set ordering, and
batch-size-32 batching -- and checks:

  1. the validation-set size matches the pinned n = 2135;
  2. the number of empty ground-truth masks in the validation split (rho);
  3. that the all-empty-prediction batch-mean Dice reproduces
     0.7774375410222295 to float precision;
  4. that ``label_int == 0`` coincides exactly with empty masks.

Usage:
    python3 scripts/verify_siim_floor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_ROOTS, RANDOM_SEED  # noqa: E402

EXPECTED_FLOOR = 0.7774375410222295
BATCH_SIZE = 32
TEST_SIZE = 0.2


def main() -> None:
    root = DATASET_ROOTS["siim"]
    index_csv = root / "preprocessed" / "index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(index_csv)

    df = pd.read_csv(index_csv)
    required = {"image_path", "mask_path", "label_int", "group_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"index.csv missing columns: {required - set(df.columns)}")

    img_dir = root / "preprocessed" / "images"
    mask_dir = root / "preprocessed" / "masks"
    df["image_path"] = df["image_path"].astype(str).map(lambda p: str(img_dir / Path(p).name))
    df["mask_path"] = df["mask_path"].astype(str).map(lambda p: str(mask_dir / Path(p).name))

    valid = df["image_path"].map(lambda p: Path(p).exists()) & df["mask_path"].map(
        lambda p: Path(p).exists()
    )
    df = df[valid].drop_duplicates(subset=["group_id"], keep="last").reset_index(drop=True)
    print(f"replicated parse_siim: n = {len(df)} (log pin: 10675)")
    if len(df) != 10675:
        print("WARNING: replicated dataset size differs from the pinned 10,675")

    labels = df["label_int"].astype(np.int64).to_numpy()
    groups = df["group_id"].astype(str).to_numpy()

    # make_group_split fallback branch: every group unique -> StratifiedShuffleSplit
    if len(np.unique(groups)) == len(labels):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        (train_idx, val_idx), = splitter.split(np.zeros(len(labels)), labels)
        split_kind = "StratifiedShuffleSplit (unique-group fallback)"
    else:
        from sklearn.model_selection import GroupShuffleSplit

        (train_idx, val_idx), = list(
            GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED).split(
                np.zeros(len(labels)), labels, groups
            )
        )
        split_kind = "GroupShuffleSplit"
    print(f"split: {split_kind} (seed={RANDOM_SEED}, test_size={TEST_SIZE})")
    print(f"val n = {len(val_idx)} (log pin: 2135)")

    val_labels = labels[val_idx]
    val_masks = df["mask_path"].to_numpy()[val_idx]

    empty_flags = []
    for path in val_masks:
        arr = np.array(Image.open(path).convert("L"))
        empty_flags.append(int(arr.sum()) == 0)
    empty_flags = np.array(empty_flags, dtype=bool)

    n_empty = int(empty_flags.sum())
    label_zero = val_labels == 0
    print(f"empty ground-truth masks in val: {n_empty}/{len(val_idx)} "
          f"(rho = {n_empty / len(val_idx):.6f})")
    print(f"label_int == 0 in val:           {int(label_zero.sum())}/{len(val_idx)} "
          f"({'EXACT match with empty masks' if (label_zero == empty_flags).all() else 'MISMATCH with empty masks'})")

    # All-empty-prediction batch-mean Dice under dice_coefficient(empty_score=1.0):
    # per-sample score = 1.0 if GT empty else 0.0; DataLoader(shuffle=False) preserves
    # dataset order; np.mean averages the 67 batch means with equal weight.
    batch_means = []
    for start in range(0, len(val_idx), BATCH_SIZE):
        chunk = empty_flags[start:start + BATCH_SIZE]
        batch_means.append(float(chunk.mean()))
    floor = float(np.mean(batch_means))
    print(f"\nbatch structure: {len(batch_means)} batches "
          f"({len(batch_means) - 1} x {BATCH_SIZE} + 1 x {len(val_idx) - 66 * BATCH_SIZE})")
    print(f"all-empty-prediction macro Dice: {floor!r}")
    print(f"checkpoint artifact value:       {EXPECTED_FLOOR!r}")
    print(f"byte-identical reproduction:     {floor == EXPECTED_FLOOR}")

    fg = float((~empty_flags).sum())
    print(f"\nimplied foreground Dice of the collapsed model on positive slices: "
          f"{(floor * len(val_idx) - n_empty) / fg:.6f} (mechanism prediction: 0.0)")


if __name__ == "__main__":
    main()
