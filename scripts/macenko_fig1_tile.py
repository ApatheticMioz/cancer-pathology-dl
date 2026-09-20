"""Regenerate the single Macenko-normalized tile consumed by Figure 1.

The original ``preprocessed_macenko_fixed/`` tree was produced out of band and
lost with the local ``data/`` purge (AUDIT_MANIFEST F-20). This driver rebuilds
the one file referenced by ``paper/figures/fig1_macenko.py`` (``_MAC_TILE``),
reusing :mod:`src.apply_macenko` unchanged:

1. a deterministic seed-42 reference is estimated from 32 sampled PANDA tiles
   via :func:`compute_gold_standard_reference` (population reference — a
   self-referenced single tile would make panel (c)'s discrepancy map
   degenerate);
2. the pinned raw tile is normalized via :func:`macenko_normalize_single`.

Usage:
    python scripts/macenko_fig1_tile.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from apply_macenko import (  # noqa: E402
    RANDOM_SEED,
    compute_gold_standard_reference,
    discover_images,
    macenko_normalize_single,
    get_panda_raw_images,
)

RAW_DIR = REPO_ROOT / "data" / "PANDA" / "train_images" / "train_images"
TILE_ID = "3dab3238ef15a3c5b3d43e0b777073a5"
RAW_TILE = RAW_DIR / f"{TILE_ID}.png"
OUT_TILE = REPO_ROOT / "data" / "PANDA" / "preprocessed_macenko_fixed" / "images" / f"{TILE_ID}.png"
REFERENCE_SAMPLE_N = 32

logger = logging.getLogger("macenko_fig1_tile")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    all_paths = get_panda_raw_images(REPO_ROOT / "data" / "PANDA")
    if RAW_TILE not in all_paths:
        all_paths.append(RAW_TILE)
    from apply_macenko import _is_mostly_white
    sample: list[Path] = []
    for path in sorted(all_paths):
        if len(sample) >= REFERENCE_SAMPLE_N:
            break
        arr = np.array(Image.open(path).convert("RGB"))
        if not _is_mostly_white(arr):
            sample.append(path)
    logger.info("estimating reference from %d tissue-bearing tiles "
                "(pool %d, seed %d)", len(sample), len(all_paths), RANDOM_SEED)
    ref_matrix, ref_conc = compute_gold_standard_reference(sample, seed=RANDOM_SEED)
    logger.info("reference stain matrix:\n%s\nreference max conc: %s",
                np.round(ref_matrix, 4), np.round(ref_conc, 4))

    img_raw = np.array(Image.open(RAW_TILE).convert("RGB"))
    img_mac = macenko_normalize_single(img_raw, ref_matrix, ref_conc)

    OUT_TILE.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_mac, mode="RGB").save(OUT_TILE)

    for name, arr in (("raw", img_raw), ("macenko", img_mac)):
        means = arr.reshape(-1, 3).mean(axis=0)
        logger.info("%s mean RGB: %s", name, np.round(means, 2))
    logger.info("wrote %s", OUT_TILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
