"""Rebuild the three pinned SIIM-ACR tiles consumed by Figure 2.

The original campaign produced ``data/SIIM/preprocessed/{images,masks}/`` out
of band; that pipeline was never committed and the artifacts were lost with the
local ``data/`` tree (AUDIT_MANIFEST F-20). This script restores *only* the
three files referenced by ``paper/figures/fig2_empty_dice.py`` from the
canonical SIIM-ACR Pneumothorax Segmentation release (stage 2 DICOM images +
RLE masks), on CPU, deterministically:

    images/<negative-uid>.png          lesion-free radiograph (hero panel)
    images/<positive-uid>.png          lesion-bearing radiograph (hero panel)
    masks/<positive-uid>_mask.png      ground-truth pneumothorax mask

Pixel data are written unmodified from the DICOM ``pixel_array`` (8-bit
MONOCHROME2, 1024x1024 as distributed); masks are the standard Kaggle
column-major RLE decode. A full preprocessed/ index.csv reconstruction is out
of scope (F-20, deferred).

Usage:
    python scripts/prep_siim_fig2_tiles.py --siim-root data/SIIM/raw
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# Pinned by paper/figures/fig2_empty_dice.py — never guess alternatives.
NEGATIVE_UID = "1.2.276.0.7230010.3.1.4.8323329.1000.1517875165.878027"
POSITIVE_UID = "1.2.276.0.7230010.3.1.4.8323329.10005.1517875220.958951"

logger = logging.getLogger("prep_siim_fig2_tiles")


def rle_decode(encoded: str, shape: tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """Decode one SIIM-ACR Kaggle RLE string (column-major, 1-indexed)."""
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if not isinstance(encoded, str) or encoded.strip() in {"", "-1"}:
        return mask.reshape(shape).T  # lesion-free row: all-empty mask
    tokens = np.array(encoded.split(), dtype=np.int64)
    starts, lengths = tokens[0::2] - 1, tokens[1::2]
    for s, l in zip(starts, lengths):
        mask[s : s + l] = 1
    # Kaggle SIIM RLE runs down columns first; transpose to row-major pixels.
    return mask.reshape(shape).T


def find_dicom(siim_root: Path, uid: str) -> Path:
    """Locate <uid>.dcm under siim_root (handles the <uid>.dcm/<uid>.dcm
    directory nesting used by the stage_2_images archives)."""
    hits = sorted(siim_root.rglob(f"{uid}.dcm"))
    if not hits:
        raise FileNotFoundError(f"{uid}.dcm not found under {siim_root}")
    return hits[-1]


def save_image(dicom_path: Path, out_path: Path) -> tuple[int, int]:
    ds = pydicom.dcmread(str(dicom_path))
    arr = ds.pixel_array
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float64)
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1.0) * 255.0
        arr = arr.astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(out_path)
    return arr.shape


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--siim-root", type=Path, required=True,
                   help="Directory containing stage_2 DICOMs and stage_2_train.csv")
    p.add_argument("--out-root", type=Path, default=REPO_ROOT / "data" / "SIIM" / "preprocessed",
                   help="Output preprocessed/ root (default: data/SIIM/preprocessed)")
    args = p.parse_args()

    rle_csv = next((f for f in args.siim_root.rglob("*.csv")
                    if f.name in {"stage_2_train.csv", "train-rle.csv"}), None)
    if rle_csv is None:
        logger.error("RLE sheet (stage_2_train.csv / train-rle.csv) not found under %s", args.siim_root)
        return 1
    df = pd.read_csv(rle_csv)
    df.columns = [c.strip() for c in df.columns]
    df["ImageId"] = df["ImageId"].astype(str).str.strip()

    img_dir, mask_dir = args.out_root / "images", args.out_root / "masks"

    shape = save_image(find_dicom(args.siim_root, NEGATIVE_UID), img_dir / f"{NEGATIVE_UID}.png")
    logger.info("negative tile written (%dx%d)", shape[1], shape[0])

    shape = save_image(find_dicom(args.siim_root, POSITIVE_UID), img_dir / f"{POSITIVE_UID}.png")
    logger.info("positive tile written (%dx%d)", shape[1], shape[0])

    row = df.loc[df["ImageId"] == POSITIVE_UID, "EncodedPixels"]
    if row.empty:
        logger.error("positive UID absent from %s", rle_csv)
        return 1
    mask = rle_decode(row.iloc[0], shape)
    if mask.sum() == 0:
        logger.error("positive UID decoded to an empty mask — wrong row?")
        return 1
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(mask_dir / f"{POSITIVE_UID}_mask.png")
    logger.info("positive mask written (%d lesion pixels)", int(mask.sum()))

    neg_row = df.loc[df["ImageId"] == NEGATIVE_UID, "EncodedPixels"]
    neg_empty = neg_row.empty or str(neg_row.iloc[0]).strip() in {"nan", "-1", ""}
    logger.info("negative UID lesion-free per RLE sheet: %s", neg_empty)
    if not neg_empty:
        logger.warning("negative UID has an RLE — fig2 expects a lesion-free tile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
