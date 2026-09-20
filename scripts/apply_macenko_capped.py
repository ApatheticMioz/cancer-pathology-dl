#!/usr/bin/env python3
"""Populate Macenko-normalized images for BOTH histopathology datasets.

Datasets
--------
* PANDA   : 10,616 raw tiles at  data/PANDA/train_images/train_images
* PanNuke : 7,901 preprocessed tiles at data/PanNuke/preprocessed/images

Outputs (apply_macenko convention)
----------------------------------
* Normalized tiles are written to  data/<D>/preprocessed_macenko_fixed/images
  as same-size uint8 PNGs (PANDA 512x512, PanNuke 256x256).
* A SYMLINK adapter  data/<D>/preprocessed_macenko -> preprocessed_macenko_fixed
  is created so that ``parse_panda`` / ``parse_pannuke`` (which read the bare
  ``preprocessed_macenko/images`` path) resolve to the fixed output. The
  adapter is created ONLY IF MISSING (or is a broken/empty leftover); a valid
  existing symlink or a working nested ``images`` symlink is left untouched.

Reference standard (IMPORTANT - explicit, not silent)
-----------------------------------------------------
``src/apply_macenko.py`` defaults to ``REFERENCE_SAMPLE_FRACTION = 1.0``
(i.e. build the gold-standard reference from 100% of the dataset). That is
far too large for a 10GB-RAM WSL box, so this script instead builds the
reference from a DETERMINISTIC UNIFORM SAMPLE capped at
``REFERENCE_CAP = 512`` tissue-bearing tiles per dataset (seed 42).

The reference is computed by ``compute_tissue_reference`` below, which
reuses the Macenko core helpers (``_optical_density``,
``_extract_stain_vectors``) from ``src/apply_macenko.py`` but selects
reference tiles by the *tissue-bearing* criterion (>= ``MIN_TISSUE_PIXELS``
pixels with OD > ``OD_TISSUE_THRESHOLD``) rather than the ``>50% white``
pre-filter. The white filter is dropped because PANDA tiles are 80-96% white
background with a central tissue island, so the white filter would reject
~99.9% of the dataset and leave a degenerate 1-tile reference; white bias is
already prevented because stain vectors are extracted from tissue pixels only.
The cap (512) and seed (42) are printed in the stdout summary below.

Memory discipline
-----------------
* Single-process, tile-by-tile streaming straight off disk (PIL open ->
  normalize -> save -> free). No whole-dataset arrays are ever materialized.
* An RSS watchdog prints ``getrusage`` (peak) + current VmRSS every
  ``WATCHDOG_EVERY = 256`` tiles and HARD-ABORTS if peak RSS exceeds
  ``RSS_LIMIT_GB = 6.5`` (WSL cap is 10GB).

Idempotency
-----------
Existing output PNGs are skipped, so re-running only processes what is
missing.

Usage
-----
    python scripts/apply_macenko_capped.py                     # both datasets
    python scripts/apply_macenko_capped.py --datasets panda
    python scripts/apply_macenko_capped.py --limit 8 --verify  # smoke test
    python scripts/apply_macenko_capped.py --verify            # parse check only
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Reuse the canonical Macenko core + discovery helpers (do NOT reinvent).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from apply_macenko import (  # noqa: E402
    macenko_normalize_single,
    get_panda_raw_images,
    get_pannuke_raw_images,
    # Core Macenko helpers reused by the tissue-bearing reference below.
    _optical_density,
    _extract_stain_vectors,
    OD_TISSUE_THRESHOLD,
    MIN_TISSUE_PIXELS,
)

logger = logging.getLogger("macenko_capped")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
REFERENCE_CAP = 512          # deterministic uniform sample cap (tissue-bearing)
RANDOM_SEED = 42
WATCHDOG_EVERY = 256        # print RSS every N tiles
RSS_LIMIT_GB = 6.5          # hard abort threshold (WSL cap is 10GB)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# RSS watchdog
# ---------------------------------------------------------------------------
def _peak_rss_gb() -> float:
    """Peak RSS of this process in GB (getrusage; ru_maxrss is KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _current_rss_gb() -> float:
    """Current VmRSS in GB from /proc/self/status (falls back to peak)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except Exception:
        pass
    return _peak_rss_gb()


def _watchdog(dataset: str, done: int, total: int) -> None:
    peak = _peak_rss_gb()
    cur = _current_rss_gb()
    logger.info(
        "[RSS] %-8s %d/%d tiles | peak=%.2fGB cur=%.2fGB (limit %.1fGB)",
        dataset.upper(), done, total, peak, cur, RSS_LIMIT_GB,
    )
    if peak > RSS_LIMIT_GB:
        logger.error(
            "[RSS] HARD ABORT: peak RSS %.2fGB exceeds limit %.1fGB "
            "(WSL cap 10GB). Stopping to protect the box.",
            peak, RSS_LIMIT_GB,
        )
        raise MemoryError(f"RSS limit exceeded: {peak:.2f}GB > {RSS_LIMIT_GB}GB")


# ---------------------------------------------------------------------------
# Deterministic reference sampling
# ---------------------------------------------------------------------------
def select_reference_sample(paths: list[Path], cap: int = REFERENCE_CAP,
                            seed: int = RANDOM_SEED) -> list[Path]:
    """Deterministic uniform sample of at most *cap* tiles (seed *seed*).

    If the dataset has <= cap tiles, all of them are used. Otherwise a
    reproducible uniform sample of exactly *cap* tiles is drawn from the
    sorted path list.
    """
    paths = sorted(paths)
    if len(paths) <= cap:
        return paths
    rng = random.Random(seed)
    return rng.sample(paths, cap)


def compute_tissue_reference(image_paths: list[Path],
                             seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Gold-standard reference from TISSUE-BEARING tiles (reuses Macenko core).

    This mirrors ``apply_macenko.compute_gold_standard_reference`` but selects
    reference tiles by the *tissue-bearing* criterion the task specifies
    (>= ``MIN_TISSUE_PIXELS`` pixels with OD > ``OD_TISSUE_THRESHOLD``) instead
    of the ``>50% white background`` pre-filter.

    Why the white filter is dropped here: PANDA tiles are 80-96% white
    background with a central tissue island, so the ``>50% white`` filter
    rejects ~99.9% of the dataset and leaves a degenerate 1-tile reference.
    The white-bias concern the filter was meant to address is already handled
    because the stain vectors are extracted from TISSUE pixels only
    (``_extract_stain_vectors`` on the OD>threshold subset), so white
    background never enters the reference. The same tissue filter is applied
    to PanNuke (harmless there, where most tiles also pass the white filter).

    Reuses ``_optical_density`` and ``_extract_stain_vectors`` from
    ``src/apply_macenko.py``; only the tile-selection predicate differs.

    Returns:
        (2x3 stain_matrix, 2-element max_conc) as float64.
    """
    rng = random.Random(seed)
    sample = select_reference_sample(image_paths, REFERENCE_CAP, seed)
    logger.info("Reference: using %d/%d tissue-bearing tiles (cap=%d, seed=%d)",
                len(sample), len(image_paths), REFERENCE_CAP, seed)

    all_stains: list[np.ndarray] = []
    all_concs: list[np.ndarray] = []
    skipped_low_tissue = 0
    skipped_corrupt = 0

    for idx, p in enumerate(sample):
        try:
            img = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            logger.warning("Reference: corrupt image %s: %s", p.name, e)
            skipped_corrupt += 1
            continue

        od = _optical_density(img)
        od_flat = od.reshape(-1, 3)
        tissue = od_flat[np.all(od_flat > OD_TISSUE_THRESHOLD, axis=1)]

        if tissue.shape[0] < MIN_TISSUE_PIXELS:
            skipped_low_tissue += 1
            continue

        stains = _extract_stain_vectors(tissue)
        concs = np.linalg.lstsq(stains, od_flat.T, rcond=None)[0]
        concs = np.maximum(concs, 0.0)
        max_c = np.percentile(concs, 99.0, axis=1)

        all_stains.append(stains)
        all_concs.append(max_c)

        if (idx + 1) % 100 == 0:
            logger.info(
                "Reference: processed %d/%d sampled images "
                "(%d valid, %d low-tissue, %d corrupt)",
                idx + 1, len(sample), len(all_stains),
                skipped_low_tissue, skipped_corrupt,
            )

    if not all_stains:
        raise RuntimeError(
            f"No tissue-bearing images for reference. Checked {len(sample)} sampled, "
            f"{skipped_low_tissue} low-tissue, {skipped_corrupt} corrupt."
        )

    stain_matrix = np.median(np.array(all_stains), axis=0)
    max_conc = np.median(np.array(all_concs), axis=0)

    logger.info(
        "Reference complete: %d tissue-bearing images used. "
        "Stain matrix:\\n%s\\nMax concentrations: %s",
        len(all_stains),
        np.array2string(stain_matrix, precision=6),
        np.array2string(max_conc, precision=6),
    )
    return stain_matrix.astype(np.float64), max_conc.astype(np.float64)


# ---------------------------------------------------------------------------
# Symlink adapter: preprocessed_macenko -> preprocessed_macenko_fixed
# ---------------------------------------------------------------------------
def ensure_adapter(root: Path) -> str:
    """Ensure ``root/preprocessed_macenko`` resolves to ``preprocessed_macenko_fixed``.

    ``parse_panda`` / ``parse_pannuke`` read the bare ``preprocessed_macenko/images``
    path, so we expose the fixed output through that name. We only create the
    symlink if it is missing; a valid existing symlink (or a working nested
    ``images`` symlink, as PANDA already has) is left untouched. A broken/empty
    leftover directory is replaced by the symlink.
    """
    fixed = root / "preprocessed_macenko_fixed"
    link = root / "preprocessed_macenko"

    if link.is_symlink():
        if Path(os.path.realpath(link)) == Path(os.path.realpath(fixed)):
            return "existing-symlink"
        link.unlink()
        os.symlink(fixed, link)
        return "recreated-symlink"

    if link.is_dir():
        nested = link / "images"
        if nested.is_symlink() and Path(os.path.realpath(nested)) == \
                Path(os.path.realpath(fixed / "images")):
            return "existing-nested-symlink"
        # Broken/empty leftover dir (no regular files) -> safe to replace.
        files = [p for p in link.rglob("*") if p.is_file()]
        if not files:
            shutil.rmtree(link)
            os.symlink(fixed, link)
            return "replaced-empty-dir-with-symlink"
        # Non-empty real dir: never clobber; add nested images symlink if missing.
        if not nested.exists():
            os.symlink(fixed / "images", nested)
            return "added-nested-symlink"
        return "left-untouched"

    # Missing -> create the symlink adapter.
    os.symlink(fixed, link)
    return "created-symlink"


# ---------------------------------------------------------------------------
# Per-dataset pipeline (single-process, tile-by-tile streaming)
# ---------------------------------------------------------------------------
def process_dataset(dataset: str, base_dir: Path, raw_images: list[Path],
                    output_dir: Path, limit: int | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("PROCESSING: %s", dataset.upper())
    logger.info("  Raw images:   %d", len(raw_images))
    logger.info("  Output dir:   %s", output_dir)
    logger.info("  Reference:    deterministic uniform sample, cap=%d, seed=%d",
                REFERENCE_CAP, RANDOM_SEED)
    logger.info("=" * 72)

    # Step 1: reference from a capped deterministic sample of tissue-bearing
    # tiles (reuses the Macenko core helpers from src/apply_macenko.py).
    stain_matrix, max_conc = compute_tissue_reference(raw_images, seed=RANDOM_SEED)
    ref_matrix = stain_matrix.astype(np.float64)
    ref_conc = max_conc.astype(np.float64)

    # Step 2: stream tiles one at a time (idempotent).
    if limit is not None:
        work = raw_images[:limit]
    else:
        work = raw_images

    ok = skipped = errors = 0
    failed: list[str] = []
    start = time.time()

    for i, img_path in enumerate(work, start=1):
        out_path = output_dir / (img_path.stem + ".png")
        if out_path.exists():
            skipped += 1
        else:
            try:
                img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
                normalized = macenko_normalize_single(img, ref_matrix, ref_conc)
                Image.fromarray(normalized, "RGB").save(str(out_path), "PNG")
                ok += 1
            except Exception as e:  # noqa: BLE001
                errors += 1
                failed.append(f"{img_path.name}: {e}")
            # release per-tile arrays promptly
            del img, normalized

        if i % WATCHDOG_EVERY == 0 or i == len(work):
            _watchdog(dataset, i, len(work))

    elapsed = time.time() - start
    logger.info(
        "[%s] DONE: ok=%d skipped=%d errors=%d in %s (%.2f img/s) peakRSS=%.2fGB",
        dataset.upper(), ok, skipped, errors,
        str(__import__("datetime").timedelta(seconds=int(elapsed))),
        (ok) / elapsed if elapsed > 0 else 0.0, _peak_rss_gb(),
    )
    if failed:
        logger.warning("[%s] first failures: %s", dataset.upper(), failed[:10])

    return {
        "dataset": dataset,
        "total": len(raw_images),
        "processed": ok,
        "skipped": skipped,
        "errors": errors,
        "elapsed_sec": round(elapsed, 2),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "ref_sample": min(len(raw_images), REFERENCE_CAP),
    }


# ---------------------------------------------------------------------------
# End-to-end verification: parse_panda / parse_pannuke (skip_macenko=False)
# ---------------------------------------------------------------------------
def verify_parsers(project_root: Path) -> None:
    # data.py imports `from src.config import ...`, so the project root must
    # be importable (in addition to the src/ dir added at module import time).
    sys.path.insert(0, str(project_root))
    from data import parse_panda, parse_pannuke

    logger.info("=" * 72)
    logger.info("END-TO-END VERIFY (skip_macenko=False)")
    logger.info("=" * 72)

    panda_root = project_root / "data" / "PANDA"
    if panda_root.is_dir():
        try:
            p = parse_panda(panda_root, skip_macenko=False)
            logger.info("  parse_panda   OK: %d rows", len(p["images"]))
        except Exception as e:  # noqa: BLE001
            logger.error("  parse_panda   FAILED: %s", e)

    pannuke_root = project_root / "data" / "PanNuke"
    if pannuke_root.is_dir():
        try:
            n = parse_pannuke(pannuke_root, skip_macenko=False)
            logger.info("  parse_pannuke OK: %d rows", len(n["images"]))
        except Exception as e:  # noqa: BLE001
            logger.error("  parse_pannuke FAILED: %s", e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Capped, memory-safe Macenko normalization for PANDA + PanNuke"
    )
    p.add_argument("--datasets", nargs="+", default=["panda", "pannuke"],
                   choices=["panda", "pannuke"])
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N tiles (smoke test).")
    p.add_argument("--verify", action="store_true",
                   help="Run parse_panda/parse_pannuke dry-load check.")
    p.add_argument("--skip-normalize", action="store_true",
                   help="Only run verification (no normalization).")
    p.add_argument("--project-root", type=Path, default=None)
    return p


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_parser().parse_args()
    project_root = args.project_root or Path(__file__).resolve().parent.parent
    data_root = project_root / "data"
    if not data_root.is_dir():
        logger.error("Data root not found: %s", data_root)
        return 1

    configs = {
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
    stats: list[dict] = []

    for ds in args.datasets:
        cfg = configs[ds]
        base_dir = cfg["base_dir"]
        if not base_dir.is_dir():
            logger.error("Dataset dir missing: %s", base_dir)
            continue

        logger.info("Discovering images for %s ...", ds.upper())
        raw = cfg["get_images"](base_dir)
        logger.info("Found %d raw images for %s", len(raw), ds.upper())

        if not args.skip_normalize:
            out_dir = base_dir / cfg["output_subdir"]
            s = process_dataset(ds, base_dir, raw, out_dir, limit=args.limit)
            stats.append(s)

        # Symlink adapter (idempotent; only created if missing/broken).
        adapter = ensure_adapter(base_dir)
        logger.info("[%s] adapter preprocessed_macenko -> preprocessed_macenko_fixed: %s",
                    ds.upper(), adapter)

    total_elapsed = time.time() - overall_start

    logger.info("")
    logger.info("=" * 72)
    logger.info("SUMMARY")
    logger.info("=" * 72)
    logger.info("Reference: deterministic UNIFORM sample, cap=%d tissue-bearing "
                "tiles/dataset, seed=%d (src default REFERENCE_SAMPLE_FRACTION=1.0 "
                "is intentionally NOT used here).", REFERENCE_CAP, RANDOM_SEED)
    for s in stats:
        logger.info(
            "  %-8s total=%-6d processed=%-6d skipped=%-6d errors=%-4d "
            "ref=%-4d time=%ss peakRSS=%.2fGB",
            s["dataset"].upper(), s["total"], s["processed"], s["skipped"],
            s["errors"], s["ref_sample"], s["elapsed_sec"], s["peak_rss_gb"],
        )
    logger.info("  Total wall time: %s",
                str(__import__("datetime").timedelta(seconds=int(total_elapsed))))
    logger.info("  Overall peak RSS: %.2fGB", _peak_rss_gb())
    logger.info("=" * 72)

    if args.verify:
        verify_parsers(project_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
