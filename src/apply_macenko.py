#!/usr/bin/env python3
"""Offline Macenko stain normalization with a robust population reference.

Replaces the corrupted preprocessed_macenko/ output with a correct
preprocessed_macenko_fixed/ output for PANDA and PanNuke datasets.

Key fixes over the previous offline script:
    - Gold Standard reference built from the MEDIAN stain matrix of 100%
      random samples, NOT a single biased reference image.
    - Images with >50% white background are excluded from the reference
      computation to prevent white-space bias.
    - Multiprocessing via ProcessPoolExecutor for CPU-bound normalization.
    - Safe output to preprocessed_macenko_fixed/images/ (does not touch
      the old preprocessed_macenko/ directory).

Usage:
    python src/offline_macenko_fixed.py
    python src/offline_macenko_fixed.py --datasets panda pannuke --workers 12
    python src/offline_macenko_fixed.py --datasets panda --skip-reference
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger("macenko_fixed")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
RANDOM_SEED = 42
REFERENCE_SAMPLE_FRACTION = 1.0
WHITE_PIXEL_THRESHOLD = 240.0
WHITE_BACKGROUND_FRACTION = 0.50
OD_TISSUE_THRESHOLD = 0.15
MIN_TISSUE_PIXELS = 1000

# ---------------------------------------------------------------------------
# Macenko core (self-contained for pickling across process boundaries)
# ---------------------------------------------------------------------------

def _optical_density(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (uint8 or float) to optical density space."""
    img = np.clip(rgb.astype(np.float32), 1.0, 255.0)
    return -np.log(img / 255.0)


def _extract_stain_vectors(od_flat: np.ndarray) -> np.ndarray:
    """Extract two dominant stain vectors via eigen-decomposition.

    Args:
        od_flat: (N, 3) OD-space pixels.

    Returns:
        2x3 normalized stain vector matrix.
    """
    cov = np.cov(od_flat, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order[:2]]

    projected = od_flat @ basis
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    low_angle, high_angle = np.percentile(angles, [1.0, 99.0])

    v1 = basis @ np.array([np.cos(low_angle), np.sin(low_angle)], dtype=np.float32)
    v2 = basis @ np.array([np.cos(high_angle), np.sin(high_angle)], dtype=np.float32)
    stains = np.stack([v1, v2], axis=1)
    stains = stains / np.linalg.norm(stains, axis=0, keepdims=True).clip(min=1e-8)
    return stains


def _is_mostly_white(img_array: np.ndarray, threshold: float = WHITE_PIXEL_THRESHOLD) -> bool:
    """Check if >50% of pixels are white (luminance > threshold on all channels)."""
    white_mask = np.all(img_array >= threshold, axis=-1)
    white_fraction = white_mask.sum() / white_mask.size
    return white_fraction > WHITE_BACKGROUND_FRACTION


def compute_gold_standard_reference(
    image_paths: list[Path],
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a robust Gold Standard stain reference.

    1. Sample 50% of images randomly.
    2. Filter out images with >50% white background.
    3. Extract Macenko stain vectors for each valid image.
    4. Return the MEDIAN stain matrix and MEDIAN max concentrations.

    Args:
        image_paths: All image paths in the dataset.
        seed: Random seed for reproducibility.

    Returns:
        (2x3 stain_matrix, 2-element max_conc).
    """
    rng = random.Random(seed)
    sample_size = max(1, int(len(image_paths) * REFERENCE_SAMPLE_FRACTION))
    sampled = rng.sample(sorted(image_paths), sample_size)

    logger.info(
        "Reference: sampling %d/%d images (seed=%d)",
        sample_size, len(image_paths), seed,
    )

    all_stains: list[np.ndarray] = []
    all_concs: list[np.ndarray] = []
    skipped_white = 0
    skipped_corrupt = 0

    for idx, p in enumerate(sampled):
        try:
            img = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
        except Exception as e:
            logger.warning("Reference: corrupt image %s: %s", p.name, e)
            skipped_corrupt += 1
            continue

        if _is_mostly_white(img):
            skipped_white += 1
            continue

        od = _optical_density(img)
        od_flat = od.reshape(-1, 3)
        tissue = od_flat[np.all(od_flat > OD_TISSUE_THRESHOLD, axis=1)]

        if tissue.shape[0] < MIN_TISSUE_PIXELS:
            continue

        stains = _extract_stain_vectors(tissue)
        concs = np.linalg.lstsq(stains, od.reshape(-1, 3).T, rcond=None)[0]
        concs = np.maximum(concs, 0.0)
        max_c = np.percentile(concs, 99.0, axis=1)

        all_stains.append(stains)
        all_concs.append(max_c)

        if (idx + 1) % 100 == 0:
            logger.info(
                "Reference: processed %d/%d sampled images "
                "(%d valid, %d white, %d corrupt)",
                idx + 1, len(sampled),
                len(all_stains), skipped_white, skipped_corrupt,
            )

    if not all_stains:
        raise RuntimeError(
            f"No valid images for reference. Checked {len(sampled)} sampled, "
            f"{skipped_white} white, {skipped_corrupt} corrupt."
        )

    stain_matrix = np.median(np.array(all_stains), axis=0)
    max_conc = np.median(np.array(all_concs), axis=0)

    logger.info(
        "Reference complete: %d valid images used. "
        "Stain matrix:\n%s\nMax concentrations: %s",
        len(all_stains),
        np.array2string(stain_matrix, precision=6),
        np.array2string(max_conc, precision=6),
    )
    return stain_matrix.astype(np.float64), max_conc.astype(np.float64)


def macenko_normalize_single(
    image: np.ndarray,
    ref_stain_matrix: np.ndarray,
    ref_max_conc: np.ndarray,
) -> np.ndarray:
    """Normalize a single RGB image against the Gold Standard reference.

    Args:
        image: RGB uint8 array (H, W, 3).
        ref_stain_matrix: 2x3 reference stain vectors.
        ref_max_conc: 2-element reference max concentrations.

    Returns:
        Normalized RGB uint8 array.
    """
    img = np.clip(image.astype(np.float32), 1.0, 255.0)
    od = _optical_density(img)

    concentrations = np.linalg.lstsq(
        ref_stain_matrix, od.reshape(-1, 3).T, rcond=None
    )[0]
    concentrations = np.maximum(concentrations, 0.0)

    denom = np.percentile(concentrations, 99.0, axis=1).clip(min=1e-6)
    scale = ref_max_conc / denom
    concentrations = concentrations * scale[:, None]

    recon = np.exp(-(ref_stain_matrix @ concentrations)).T.reshape(img.shape)
    return np.clip(recon * 255.0, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Multiprocessing worker (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _normalize_worker(
    image_path: str,
    output_path: str,
    ref_stain_matrix: np.ndarray,
    ref_max_conc: np.ndarray,
) -> dict:
    """Worker function for ProcessPoolExecutor.

    Normalizes one image and saves it to the output path.

    Returns:
        Dict with status info for logging.
    """
    try:
        img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
        normalized = macenko_normalize_single(img, ref_stain_matrix, ref_max_conc)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(normalized, "RGB").save(str(out), "PNG")
        return {"status": "ok", "file": Path(image_path).name}
    except Exception as e:
        return {"status": "error", "file": Path(image_path).name, "error": str(e)}


# ---------------------------------------------------------------------------
# Dataset image discovery
# ---------------------------------------------------------------------------

def discover_images(directory: Path) -> list[Path]:
    """Recursively find all image files under *directory*."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(directory.rglob(f"*{ext}"))
    images = [p for p in images if p.is_file()]
    images.sort()
    return images


def get_panda_raw_images(base_dir: Path) -> list[Path]:
    """Locate PANDA raw images.

    Checks train_images/train_images/ first (nested structure),
    then falls back to train_images/ and images/.
    """
    candidates = [
        base_dir / "train_images" / "train_images",
        base_dir / "train_images",
        base_dir / "images",
    ]
    for d in candidates:
        if d.is_dir():
            imgs = discover_images(d)
            if imgs:
                return imgs
    raise FileNotFoundError(f"No PANDA raw images found under {base_dir}")


def get_pannuke_raw_images(base_dir: Path) -> list[Path]:
    """Locate PanNuke raw images from preprocessed/images/."""
    img_dir = base_dir / "preprocessed" / "images"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"PanNuke preprocessed/images missing: {img_dir}")
    imgs = discover_images(img_dir)
    if not imgs:
        raise FileNotFoundError(f"No PanNuke images found in {img_dir}")
    return imgs


# ---------------------------------------------------------------------------
# Dataset processing pipeline
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_name: str,
    base_dir: Path,
    raw_images: list[Path],
    output_dir: Path,
    workers: int,
    skip_reference: bool = False,
    reference_cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the full normalization pipeline for one dataset.

    Args:
        dataset_name: Human-readable name (e.g. 'PANDA').
        base_dir: Dataset root directory.
        raw_images: List of raw image paths.
        output_dir: Target output directory.
        workers: Number of parallel workers.
        skip_reference: If True, reuse reference from *reference_cache*.
        reference_cache: Dict with 'stain_matrix' and 'max_conc' keys.

    Returns:
        (stain_matrix, max_conc, stats_dict).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PROCESSING: %s", dataset_name.upper())
    logger.info("  Raw images:    %d", len(raw_images))
    logger.info("  Output dir:    %s", output_dir)
    logger.info("  Workers:       %d", workers)
    logger.info("=" * 70)

    # Step 1: Compute or reuse Gold Standard reference
    if skip_reference and reference_cache:
        stain_matrix = reference_cache["stain_matrix"]
        max_conc = reference_cache["max_conc"]
        logger.info("Using cached reference (skip_reference=True)")
    else:
        stain_matrix, max_conc = compute_gold_standard_reference(raw_images)

    # Step 2: Build task list
    tasks = []
    for img_path in raw_images:
        out_name = img_path.stem + ".png"
        out_path = output_dir / out_name
        if out_path.exists():
            tasks.append((str(img_path), str(out_path), "skip"))
        else:
            tasks.append((str(img_path), str(out_path), "process"))

    to_process = [(ip, op) for ip, op, action in tasks if action == "process"]
    already_done = sum(1 for _, _, a in tasks if a == "skip")

    logger.info(
        "Tasks: %d total, %d to process, %d already exist",
        len(tasks), len(to_process), already_done,
    )

    if not to_process:
        logger.info("All images already normalized. Skipping.")
        return stain_matrix, max_conc, {
            "dataset": dataset_name,
            "total": len(raw_images),
            "processed": 0,
            "skipped": already_done,
            "errors": 0,
        }

    # Step 3: Multiprocessing normalization
    start_time = time.time()
    stats = {"ok": 0, "error": 0, "skipped": already_done}
    failed_files: list[str] = []

    ref_matrix_f64 = stain_matrix.astype(np.float64)
    ref_conc_f64 = max_conc.astype(np.float64)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for img_path, out_path in to_process:
            fut = executor.submit(
                _normalize_worker, img_path, out_path,
                ref_matrix_f64, ref_conc_f64,
            )
            futures[fut] = img_path

        total = len(futures)
        completed = 0

        for fut in as_completed(futures):
            completed += 1
            result = fut.result()
            if result["status"] == "ok":
                stats["ok"] += 1
            else:
                stats["error"] += 1
                failed_files.append(result["file"])

            if completed % 500 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                logger.info(
                    "[%s] %d/%d images (%.1f img/s, ETA %ds) "
                    "ok=%d err=%d",
                    dataset_name.upper(), completed, total,
                    rate, int(eta), stats["ok"], stats["error"],
                )

    elapsed = time.time() - start_time
    logger.info(
        "[%s] DONE: %d processed in %s (%.1f img/s). Errors: %d",
        dataset_name.upper(), stats["ok"],
        str(__import__("datetime").timedelta(seconds=int(elapsed))),
        stats["ok"] / elapsed if elapsed > 0 else 0,
        stats["error"],
    )

    if failed_files:
        logger.warning("[%s] Failed files: %s", dataset_name.upper(), failed_files[:20])

    return stain_matrix, max_conc, {
        "dataset": dataset_name,
        "total": len(raw_images),
        "processed": stats["ok"],
        "skipped": stats["skipped"],
        "errors": stats["error"],
        "elapsed_sec": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline Macenko stain normalization with robust population reference"
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["panda", "pannuke"],
        choices=["panda", "pannuke"],
        help="Datasets to process (default: panda pannuke)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 8,
        help="Number of parallel workers (default: CPU count)",
    )
    p.add_argument(
        "--skip-reference",
        action="store_true",
        default=False,
        help="Skip reference computation and reuse cached reference",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reference sampling (default: {RANDOM_SEED})",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory (default: auto-detected)",
    )
    return p


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args()

    project_root = args.project_root or Path(__file__).resolve().parent.parent
    data_root = project_root / "data"

    if not data_root.is_dir():
        logger.error("Data root not found: %s", data_root)
        return 1

    dataset_configs = {
        "panda": {
            "base_dir": data_root / "PANDA",
            "get_images": get_panda_raw_images,
            "output_subdir": Path("preprocessed_macenko_fixed") / "images",
        },
        "pannuke": {
            "base_dir": data_root / "PanNuke",
            "get_images": get_pannuke_raw_images,
            "output_subdir": Path("preprocessed_macenko_fixed") / "images",
        },
    }

    overall_start = time.time()
    all_stats = []
    reference_cache = None

    for ds_name in args.datasets:
        cfg = dataset_configs[ds_name]
        base_dir = cfg["base_dir"]

        if not base_dir.is_dir():
            logger.error("Dataset directory missing: %s", base_dir)
            continue

        logger.info("Discovering images for %s ...", ds_name.upper())
        raw_images = cfg["get_images"](base_dir)
        logger.info("Found %d raw images for %s", len(raw_images), ds_name.upper())

        output_dir = base_dir / cfg["output_subdir"]

        stain_matrix, max_conc, stats = process_dataset(
            dataset_name=ds_name,
            base_dir=base_dir,
            raw_images=raw_images,
            output_dir=output_dir,
            workers=args.workers,
            skip_reference=args.skip_reference,
            reference_cache=reference_cache,
        )

        reference_cache = {
            "stain_matrix": stain_matrix,
            "max_conc": max_conc,
        }
        all_stats.append(stats)

    total_elapsed = time.time() - overall_start

    logger.info("")
    logger.info("=" * 70)
    logger.info("NORMALIZATION COMPLETE")
    logger.info("=" * 70)
    for s in all_stats:
        logger.info(
            "  %-10s  total=%-6d  processed=%-6d  skipped=%-6d  errors=%-4d  time=%ss",
            s["dataset"].upper(),
            s["total"], s["processed"], s["skipped"], s["errors"],
            s.get("elapsed_sec", 0),
        )
    logger.info("  Total wall time: %s", str(__import__("datetime").timedelta(seconds=int(total_elapsed))))
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())