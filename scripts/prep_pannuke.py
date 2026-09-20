#!/usr/bin/env python3
"""PanNuke mirror -> parser-contract preprocessing (memory-safe).

Converts the raw PanNuke mirror layout
    data/PanNuke/_dl/Part {1,2,3}/Images/images.npy   (N, 256, 256, 3) float64
    data/PanNuke/_dl/Part {1,2,3}/Images/types.npy    (N,) str, tissue type
    data/PanNuke/_dl/Part {1,2,3}/Masks/masks.npy     (N, 256, 256, 6) float64
into the layout consumed by ``src/data.py::parse_pannuke``:
    <out>/preprocessed/images/<patch>.png   RGB uint8
    <out>/preprocessed/masks/<patch>.png    single-channel uint8, values 0..5
    <out>/preprocessed/index.csv            columns: image_path, mask_path,
                                            label_int, group_id

Mask semantics (verified against the mirror, Part 1):
    ``masks.npy`` is INSTANCE-wise, not class-wise: each of the 6 channels
    holds per-instance integer IDs (0 = no instance, 1..K = instance ID;
    observed max ID 606 in Part 1). Channel order per the author README:
        0 Neoplastic, 1 Inflammatory, 2 Connective/Soft tissue,
        3 Dead, 4 Epithelial, 5 Background.
    Reduction to the single-channel 6-class encoding the seg loss expects
    (DATASET_META["pannuke"]["seg_classes"] == 6, CrossEntropyLoss, values
    in [0, 6) as enforced in src/training.py):
        class = first channel in 0..4 with a non-zero instance ID,
                else 5 (background).
    The 6th channel (index 5, "Background") carries no class information and
    is ignored by the reduction.

group_id / granularity:
    This mirror contains NO slide IDs -- patches are the atomic unit. The
    declared granularity is therefore the patch itself:
        group_id = f"pannuke_{global_idx:06d}"
    with a stable global index (parts in ascending order, patches in array
    order). ``parse_pannuke`` dedups on group_id (keep="last"), so group_id
    MUST be unique per patch; this construction guarantees that.

label_int:
    Tissue type -> canonical 19-class PanNuke index (num_classes=19 in
    src/config.py). The mirror's types.npy carries 18 unique names (one of
    the 19 canonical tissues is absent from this mirror); any unknown name
    fails fast.

Memory safety (WSL 10GB hard cap -- this constraint outranks speed):
    * ALL large arrays are opened with np.load(mmap_mode='r'); only
      per-patch slices (a few MB each) are ever materialized.
    * dtype/range validation is done on the FIRST chunk of each part only;
      no whole-array operations, ever.
    * RSS (ru_maxrss) is printed every 256 patches and the run hard-aborts
      if it exceeds 6.5GB.
    * Idempotent resume: already-emitted PNGs are skipped (write skipped,
      index row still recorded); index.csv is (re)written at the end.

Determinism & failure policy:
    No randomness, fixed iteration order, fixed PNG encoding. Fail fast
    (SystemExit with a message) on: missing part directory or .npy file,
    shape/length mismatch, non-integer mask values, unknown tissue name.
    No silent skips.

Usage:
    python scripts/prep_pannuke.py --parts 1,2,3            # full build
    python scripts/prep_pannuke.py --parts 1 --max-patches 64 \
        --out data/PanNuke/_prep_test                       # dry-run subset
"""
from __future__ import annotations

import argparse
import csv
import resource
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
DL_ROOT = BASE_DIR / "data" / "PanNuke" / "_dl"

# Canonical PanNuke 19-tissue schema (num_classes=19 in src/config.py).
# The mirror's types.npy uses these exact spellings (18 of the 19 present).
TISSUE_TO_CLASS = {
    "Breast": 0,
    "Colon": 1,
    "Bile-duct": 2,
    "Esophagus": 3,
    "Uterus": 4,
    "Lung": 5,
    "Cervix": 6,
    "HeadNeck": 7,
    "Skin": 8,
    "Adrenal_gland": 9,
    "Kidney": 10,
    "Stomach": 11,
    "Prostate": 12,
    "Testis": 13,
    "Liver": 14,
    "Thyroid": 15,
    "Pancreatic": 16,
    "Ovarian": 17,
    "Bladder": 18,
}

SEG_CLASSES = 6  # 5 nucleus classes + background; matches DATASET_META["pannuke"]
NUM_CLASSES = 19
PATCH_SIZE = 256
RSS_LIMIT_KB = int(6.5 * 1024 * 1024)  # hard abort threshold (ru_maxrss, KB)
RSS_REPORT_EVERY = 256
VALIDATE_CHUNK = 64  # dtype/range validation on first chunk only


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _load_part_mmap(part: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open one part's .npy triple as read-only memory maps.

    Only shape/dtype metadata is read from the file; no data is loaded.
    """
    part_dir = DL_ROOT / f"Part {part}"
    paths = {
        "images": part_dir / "Images" / "images.npy",
        "types": part_dir / "Images" / "types.npy",
        "masks": part_dir / "Masks" / "masks.npy",
    }
    for name, p in paths.items():
        if not p.exists():
            _fail(f"Part {part}: missing {p} ({name})")
    img = np.load(paths["images"], mmap_mode="r")
    typ = np.load(paths["types"], mmap_mode="r")
    msk = np.load(paths["masks"], mmap_mode="r")
    if img.ndim != 4 or img.shape[1:] != (PATCH_SIZE, PATCH_SIZE, 3):
        _fail(f"Part {part}: images.npy shape {img.shape} != (N,{PATCH_SIZE},{PATCH_SIZE},3)")
    if msk.ndim != 4 or msk.shape[1:] != (PATCH_SIZE, PATCH_SIZE, 6):
        _fail(f"Part {part}: masks.npy shape {msk.shape} != (N,{PATCH_SIZE},{PATCH_SIZE},6)")
    if not (img.shape[0] == msk.shape[0] == typ.shape[0]):
        _fail(
            f"Part {part}: length mismatch "
            f"images={img.shape[0]} masks={msk.shape[0]} types={typ.shape[0]}"
        )
    return img, typ, msk


def _validate_first_chunk(part: int, img: np.ndarray, typ: np.ndarray, msk: np.ndarray) -> None:
    """Validate dtype/range on the first chunk only (never whole-array)."""
    n = min(VALIDATE_CHUNK, img.shape[0])
    img_c = np.asarray(img[:n])
    msk_c = np.asarray(msk[:n])
    typ_c = np.asarray(typ[:n])
    if img_c.dtype != np.float64:
        _fail(f"Part {part}: images dtype {img_c.dtype} != float64")
    if msk_c.dtype != np.float64:
        _fail(f"Part {part}: masks dtype {msk_c.dtype} != float64")
    if not np.all(msk_c == np.floor(msk_c)):
        _fail(f"Part {part}: first chunk of masks.npy contains non-integer values")
    if img_c.min() < 0 or img_c.max() > 255:
        _fail(
            f"Part {part}: first chunk of images.npy out of [0,255] "
            f"(min={img_c.min()}, max={img_c.max()})"
        )
    unknown = {t for t in np.unique(typ_c) if str(t) not in TISSUE_TO_CLASS}
    if unknown:
        _fail(f"Part {part}: unknown tissue types in first chunk: {sorted(unknown)}")


def _reduce_mask(m: np.ndarray) -> np.ndarray:
    """(256,256,6) instance IDs -> (256,256) class map in [0,6).

    class = first channel in 0..4 with a non-zero instance ID, else 5
    (background). Channel 5 (Background) is ignored.
    """
    cls = np.full(m.shape[:2], SEG_CLASSES - 1, dtype=np.uint8)
    for c in range(SEG_CLASSES - 1):
        sel = (m[..., c] > 0) & (cls == SEG_CLASSES - 1)
        cls[sel] = c
    return cls


def _images_to_uint8(img: np.ndarray) -> np.ndarray:
    """float64 [0,255] RGB -> uint8 (clip guards against out-of-range values)."""
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PanNuke mirror -> parse_pannuke contract preprocessing (memory-safe)"
    )
    ap.add_argument(
        "--parts",
        default="1,2,3",
        help="comma-separated part numbers to process (default: 1,2,3)",
    )
    ap.add_argument(
        "--out",
        default=str(BASE_DIR / "data" / "PanNuke"),
        help="output root; preprocessed/{images,masks,index.csv} are created inside",
    )
    ap.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="cap on total patches emitted (dry-run subset; default: all)",
    )
    args = ap.parse_args()

    parts = sorted({int(p) for p in args.parts.split(",") if p.strip()})
    if not parts:
        _fail("--parts is empty")

    out_root = Path(args.out)
    img_dir = out_root / "preprocessed" / "images"
    mask_dir = out_root / "preprocessed" / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    per_part: dict[int, int] = {}
    label_counts: dict[int, int] = {}
    global_idx = 0
    written = 0
    resumed = 0

    for part in parts:
        img, typ, msk = _load_part_mmap(part)
        _validate_first_chunk(part, img, typ, msk)
        n = img.shape[0]
        if args.max_patches is not None:
            n = min(n, max(0, args.max_patches - global_idx))
            if n == 0:
                break
        part_rows = 0
        for i in range(n):
            name = f"pannuke_{global_idx:06d}"
            img_png = img_dir / f"{name}.png"
            mask_png = mask_dir / f"{name}.png"
            if img_png.exists() and mask_png.exists():
                resumed += 1  # idempotent resume: skip writes, keep index row
            else:
                # Per-patch slices off the mmap: ~1.6MB (img) + ~3.1MB (mask).
                Image.fromarray(_images_to_uint8(np.asarray(img[i])), mode="RGB").save(img_png)
                Image.fromarray(_reduce_mask(np.asarray(msk[i])), mode="L").save(mask_png)
                written += 1
            t = str(typ[i])
            if t not in TISSUE_TO_CLASS:
                _fail(f"Part {part} patch {i}: unknown tissue type {t!r}")
            label = TISSUE_TO_CLASS[t]
            if not 0 <= label < NUM_CLASSES:
                _fail(f"Part {part} patch {i}: label {label} outside [0,{NUM_CLASSES})")
            label_counts[label] = label_counts.get(label, 0) + 1
            rows.append(
                {
                    "image_path": f"images/{name}.png",
                    "mask_path": f"masks/{name}.png",
                    "label_int": label,
                    "group_id": name,
                }
            )
            global_idx += 1
            part_rows += 1
            if global_idx % RSS_REPORT_EVERY == 0:
                rss = _rss_gb()
                print(f"  [progress] {global_idx} patches, RSS {rss:.2f} GB", flush=True)
                if rss * 1024 > RSS_LIMIT_KB:
                    _fail(f"RSS {rss:.2f} GB exceeded 6.5GB limit -- aborting to protect WSL")
        per_part[part] = part_rows

    index_csv = out_root / "preprocessed" / "index.csv"
    with open(index_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["image_path", "mask_path", "label_int", "group_id"]
        )
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------------
    # stdout summary
    # ------------------------------------------------------------------
    print("PanNuke preprocessing summary")
    print(f"  out root      : {out_root}")
    print(f"  parts         : {parts}")
    for part in parts:
        print(f"  part {part}       : {per_part.get(part, 0)} patches")
    print(f"  total rows    : {len(rows)}")
    print(f"  written       : {written}")
    print(f"  resumed (skipped existing PNGs): {resumed}")
    groups = {r["group_id"] for r in rows}
    print(f"  unique groups : {len(groups)}")
    print("  label distribution (class: count):")
    for k in sorted(label_counts):
        print(f"    {k:2d}: {label_counts[k]}")
    print(f"  peak RSS      : {_rss_gb():.2f} GB")
    print(f"  index.csv     : {index_csv}")
    if len(rows) < 4000:
        print(
            "  NOTE: <4000 rows -> parse_pannuke's size floor will reject this "
            "index (expected for a subset dry-run)."
        )


if __name__ == "__main__":
    main()
