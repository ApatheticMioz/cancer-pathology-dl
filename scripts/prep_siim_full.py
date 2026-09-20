"""Full SIIM-ACR Pneumothorax Segmentation preprocessing builder.

Resurrects the logic of the deleted ``repro/prepare.py::_build_siim_preprocessed``
(commit 5711aaa, lines ~452-545) and adapts it to the CURRENT parser contract in
``src/data.py::parse_siim`` (lines 295-348):

    data/SIIM/preprocessed/index.csv
        columns: {image_path, mask_path, label_int, group_id}

Contract details honored here:
  * ``parse_siim`` rebuilds absolute paths from FILENAMES ONLY
    (``img_dir / Path(image_path).name``), so the CSV stores paths relative to
    the SIIM root (``preprocessed/images/<uid>.png`` /
    ``preprocessed/masks/<uid>_mask.png``); only the basename matters.
  * ``parse_siim`` dedups by ``group_id`` with ``keep="last"`` and requires
    >= 1000 surviving rows; this builder emits one row per DICOM and fails
    hard (no silent skips) if the final count is below 1000.

group_id semantics (PATIENT-level — the paper's isolation claim):
  * ``--index-only`` re-emits ``index.csv`` with ``group_id = PatientID``
    read from the DICOM header via ``pydicom.dcmread(path,
    stop_before_pixels=True)`` (headers only, never pixel arrays).
    Hierarchy:
      1. ``PatientID`` (per row);
      2. per-row fallback to ``StudyInstanceUID`` when that row's PatientID
         is missing/empty;
      3. corpus-wide fallback to ``StudyInstanceUID`` when PatientID is
         degenerate (all valid values identical);
      4. FAIL FAST with a clear error when both PatientID and
         StudyInstanceUID are degenerate.
  * The original full build emitted ``group_id = SOPInstanceUID`` (one
    trivial group per image, 10,675 groups). That made every group a
    singleton, which silently forced ``make_group_kfold_splits`` onto its
    StratifiedKFold fallback and defeated patient-level isolation. The
    full-build path is unchanged; ``--index-only`` upgrades the grouping
    in place without re-decoding pixels (image/mask path columns stay
    byte-stable).

Pixel/mask conventions (reused from the proven ``scripts/prep_siim_fig2_tiles.py``):
  * Images: raw DICOM ``pixel_array`` written unmodified when already uint8
    (1024x1024 MONOCHROME2 as distributed); min-max normalized to uint8 only
    for non-uint8 arrays. Saved as single-channel PNG (mode "L").
  * Masks: standard Kaggle SIIM RLE decode — 1-indexed, COLUMN-MAJOR runs,
    ``reshape(h, w).T`` to row-major, 0/1 values, saved as uint8 PNG
    (``mask * 255``). Multiple RLE rows for one ImageId are OR-combined
    (``np.maximum``), matching the original builder.
  * ``EncodedPixels`` of ``-1`` / ``nan`` / empty => negative: the image is
    still emitted together with an all-empty mask and ``label_int = 0``.
  * CSV whitespace: column names, ``ImageId`` and ``EncodedPixels`` values are
    stripped (the release CSV is ``"ImageId, EncodedPixels"`` with a space
    after the comma). Both CSV name variants are accepted:
    ``train-rle.csv`` and ``stage_2_train.csv``.

Fail-fast policy: an unreadable DICOM, a DICOM whose RLE row decodes to a
shape mismatch, or a final row count < 1000 raises immediately. The release
DICOM set contains 37 files whose ImageId (filename stem AND header
SOPInstanceUID) has no row in train-rle.csv; these are SKIPPED BUT REPORTED
(counted, listed in the summary, never silent) — the original builder
``continue``d on them silently, which this contract forbids. CSV rows without
a DICOM are counted and reported (the release CSV has more rows than the
DICOM set) but are not silently dropped from the summary.

Resume: DICOMs whose image+mask PNGs already exist on disk are skipped
without re-decoding (idempotent re-runs after a crash); their rows are
rebuilt from the CSV lookup so index.csv is complete.

Pinned fig2 tiles: the two hero radiographs (and the positive mask) referenced
by ``paper/figures/fig2_empty_dice.py`` are never deleted. If this builder's
deterministic output collides with an existing pinned file, the new bytes are
compared against the on-disk bytes and a mismatch is a hard error; identical
bytes are kept as-is (idempotent re-runs).

Determinism: DICOMs are processed in sorted path order; PNG bytes are computed
in memory and only written when they differ from the existing file.

Usage:
    python scripts/prep_siim_full.py \
        [--siim-root data/SIIM] \
        [--dicom-dir data/SIIM/_dl/dicom-images-train] \
        [--rle-csv data/SIIM/_dl/train-rle.csv] \
        [--index-only]
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

# Pinned by paper/figures/fig2_empty_dice.py — never delete, only regenerate
# byte-identically on collision.
PINNED_FILES = {
    "images/1.2.276.0.7230010.3.1.4.8323329.1000.1517875165.878027.png",
    "images/1.2.276.0.7230010.3.1.4.8323329.10005.1517875220.958951.png",
    "masks/1.2.276.0.7230010.3.1.4.8323329.10005.1517875220.958951_mask.png",
}

MIN_VALID_ROWS = 1000
CHECKPOINT_EVERY = 500

logger = logging.getLogger("prep_siim_full")


def rle_decode(encoded: str, shape: tuple[int, int]) -> np.ndarray:
    """Decode one SIIM-ACR Kaggle RLE string (1-indexed, column-major).

    Returns a row-major uint8 mask with 0/1 values. ``-1`` / ``nan`` / empty
    input yields an all-empty mask (negative row).
    """
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    if not isinstance(encoded, str) or encoded.strip() in {"", "-1", "nan"}:
        return mask.reshape((h, w)).T
    tokens = np.array(encoded.split(), dtype=np.int64)
    starts, lengths = tokens[0::2] - 1, tokens[1::2]
    for s, l in zip(starts, lengths):
        mask[s : s + l] = 1
    # Kaggle SIIM RLE runs down columns first; transpose to row-major pixels.
    return mask.reshape((h, w)).T


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Return the pixel array as uint8 (raw when already uint8)."""
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)


def _png_bytes(arr: np.ndarray) -> bytes:
    """Deterministic single-channel PNG bytes for a uint8 array."""
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _write_png(path: Path, data: bytes) -> str:
    """Write PNG bytes; enforce the pinned-tile invariant on collision."""
    rel = f"{path.parent.name}/{path.name}"
    if path.exists():
        existing = path.read_bytes()
        if existing == data:
            return "identical"
        if rel in PINNED_FILES:
            raise RuntimeError(
                f"pinned fig2 tile {rel} would be overwritten with different "
                f"bytes — refusing (determinism violation)"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "written"


def _load_rle_lookup(rle_csv: Path) -> dict[str, list[str]]:
    """Load ImageId -> [EncodedPixels, ...] with CSV-whitespace handling."""
    if not rle_csv.exists():
        raise FileNotFoundError(f"SIIM RLE CSV missing: {rle_csv}")
    df = pd.read_csv(rle_csv)
    df.columns = [c.strip() for c in df.columns]
    if "ImageId" not in df.columns or "EncodedPixels" not in df.columns:
        raise ValueError(f"Invalid SIIM CSV columns in {rle_csv}: {list(df.columns)}")
    df["ImageId"] = df["ImageId"].astype(str).str.strip()
    df["EncodedPixels"] = df["EncodedPixels"].astype(str).str.strip()

    lookup: dict[str, list[str]] = {}
    for image_id, encoded in zip(df["ImageId"], df["EncodedPixels"]):
        if image_id in {"", "nan"}:
            continue
        lookup.setdefault(image_id, []).append(encoded)
    if not lookup:
        raise RuntimeError(f"SIIM RLE lookup is empty: {rle_csv}")
    return lookup


def _is_degenerate(values: list[str]) -> bool:
    """True when a header field is missing/empty on every row or all identical."""
    valid = {v for v in values if v}
    return len(valid) <= 1


def reindex_patient_groups(
    siim_root: Path,
    dicom_dir: Path,
) -> dict:
    """Re-emit ``preprocessed/index.csv`` with PATIENT-level ``group_id``.

    Reads each DICOM header ONLY (``stop_before_pixels=True`` — pixel arrays
    are never decoded, so RAM stays flat) and rewrites ``group_id`` from
    ``SOPInstanceUID`` (one trivial group per image) to the patient-level
    hierarchy:

        1. ``PatientID`` (per row)
        2. ``StudyInstanceUID`` (per-row fallback when PatientID is empty)
        3. corpus-wide ``StudyInstanceUID`` (when PatientID is degenerate:
           all valid values identical)
        4. FAIL FAST when both PatientID and StudyInstanceUID are degenerate

    All other columns (``image_path``, ``mask_path``, ``label_int``) and the
    row order are byte-stable: only ``group_id`` changes. Rows whose DICOM
    cannot be read are a hard error (fail-fast, no silent skips).
    """
    index_csv = siim_root / "preprocessed" / "index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(f"SIIM preprocessed index missing: {index_csv}")
    df = pd.read_csv(index_csv)
    required = {"image_path", "mask_path", "label_int", "group_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"invalid index.csv columns: {list(df.columns)}")

    # DICOMs are nested study/series/SOP; map SOPInstanceUID stem -> path.
    stem_to_path = {p.stem: p for p in dicom_dir.rglob("*.dcm")}
    dcm_paths: list[Path] = []
    for p in df["image_path"]:
        # image_path is relative to the SIIM root; the DICOM stem is the
        # SOPInstanceUID (== the old group_id).
        dcm = stem_to_path.get(Path(p).stem)
        if dcm is None:
            raise RuntimeError(
                f"fail-fast: index row {p} has no DICOM under {dicom_dir}"
            )
        dcm_paths.append(dcm)
    missing = [str(p) for p in dcm_paths if not p.exists()]
    if missing:
        raise RuntimeError(
            f"fail-fast: {len(missing)} index rows have no DICOM under "
            f"{dicom_dir}; first: {missing[0]}"
        )

    patient_ids: list[str] = []
    study_ids: list[str] = []
    n = len(dcm_paths)
    for i, p in enumerate(dcm_paths, start=1):
        ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        pid = str(getattr(ds, "PatientID", "") or "").strip()
        sid = str(getattr(ds, "StudyInstanceUID", "") or "").strip()
        patient_ids.append(pid)
        study_ids.append(sid)
        if i % 2048 == 0:
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            logger.info("headers %d/%d read, RSS %.1f MB", i, n, rss_mb)

    use_patient = not _is_degenerate(patient_ids)
    if use_patient:
        group_ids = [pid if pid else sid for pid, sid in zip(patient_ids, study_ids)]
        source = "PatientID (per-row StudyInstanceUID fallback)"
        provenance_source = "PatientID"
    else:
        if _is_degenerate(study_ids):
            raise RuntimeError(
                "fail-fast: SIIM DICOM headers are degenerate — PatientID is "
                f"missing/identical on all {n} rows AND StudyInstanceUID is "
                "missing/identical; cannot build patient-level groups"
            )
        group_ids = study_ids
        source = "StudyInstanceUID (PatientID degenerate)"
        provenance_source = "StudyInstanceUID"

    # Rows whose per-row PatientID was empty/missing and therefore fell back to
    # StudyInstanceUID. In the degenerate branch every row uses StudyInstanceUID.
    fallback_rows_used = int(sum(1 for pid in patient_ids if not pid))

    df["group_id"] = group_ids
    df.to_csv(index_csv, index=False)

    sizes = df["group_id"].value_counts()
    n_patients = int(sizes.shape[0])

    # Emit the grouping-provenance sidecar next to the index. This is the
    # semantic record that lets the splitters (src/data.py::_validate_grouping)
    # distinguish a LEGITIMATE degenerate cardinality (a release that genuinely
    # has one image per patient, proven by source == "PatientID") from an
    # unproven one (which stays FATAL).
    provenance = {
        "source": provenance_source,
        "n_groups": int(n_patients),
        "max_group_size": int(sizes.max()),
        "fallback_rows_used": fallback_rows_used,
    }
    provenance_path = siim_root / "preprocessed" / "grouping_provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with open(provenance_path, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)
        fh.write("\n")

    summary = {
        "index_csv": str(index_csv),
        "images": int(len(df)),
        "group_source": source,
        "unique_groups": n_patients,
        "group_size_min": int(sizes.min()),
        "group_size_median": float(sizes.median()),
        "group_size_max": int(sizes.max()),
        "patients_with_gt1_image": int((sizes > 1).sum()),
        "fallback_rows_used": fallback_rows_used,
        "provenance_sidecar": str(provenance_path),
        "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }
    return summary


def build(
    siim_root: Path,
    dicom_dir: Path,
    rle_csv: Path | None = None,
) -> dict:
    """Build data/SIIM/preprocessed/{images,masks,index.csv} from DICOMs + RLE CSV."""
    if rle_csv is None:
        rle_csv = next(
            (p for p in (dicom_dir.parent / "train-rle.csv", dicom_dir.parent / "stage_2_train.csv") if p.exists()),
            None,
        )
        if rle_csv is None:
            raise FileNotFoundError(
                f"no RLE CSV (train-rle.csv / stage_2_train.csv) found next to {dicom_dir}"
            )

    lookup = _load_rle_lookup(rle_csv)

    pre_dir = siim_root / "preprocessed"
    image_dir = pre_dir / "images"
    mask_dir = pre_dir / "masks"
    index_csv = pre_dir / "index.csv"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    dcm_candidates = sorted(dicom_dir.rglob("*.dcm"))
    if not dcm_candidates:
        raise RuntimeError(f"No SIIM DICOM files found under {dicom_dir}")

    rows: list[dict] = []
    skipped: list[str] = []
    n_written = n_identical = n_resumed = 0

    for i, dcm_path in enumerate(dcm_candidates, start=1):
        image_id = dcm_path.stem
        rles = lookup.get(image_id)
        if rles is None:
            # Skip-but-report: the release set has DICOMs with no RLE-sheet row
            # (verified: neither the filename stem nor the header
            # SOPInstanceUID matches). The original builder skipped these
            # silently; this contract requires them to be counted and listed.
            skipped.append(image_id)
            continue

        safe_id = image_id  # SOPInstanceUIDs are already filesystem-safe
        image_out = image_dir / f"{safe_id}.png"
        mask_out = mask_dir / f"{safe_id}_mask.png"

        has_positive = any(r not in {"", "-1", "nan"} for r in rles)

        if image_out.exists() and mask_out.exists():
            # Resume: outputs already emitted by a previous (crashed) run —
            # skip the DICOM decode, rebuild the row from the CSV lookup.
            n_resumed += 1
        else:
            ds = pydicom.dcmread(str(dcm_path))
            px = _normalize_to_uint8(ds.pixel_array)
            h, w = px.shape

            mask = np.zeros((h, w), dtype=np.uint8)
            for encoded in rles:
                if encoded in {"", "-1", "nan"}:
                    continue
                mask = np.maximum(mask, rle_decode(encoded, (h, w)))

            img_status = _write_png(image_out, _png_bytes(px))
            mask_status = _write_png(mask_out, _png_bytes(mask * 255))
            n_written += (img_status == "written") + (mask_status == "written")
            n_identical += (img_status == "identical") + (mask_status == "identical")

        rows.append(
            {
                "image_path": f"preprocessed/images/{safe_id}.png",
                "mask_path": f"preprocessed/masks/{safe_id}_mask.png",
                "label_int": int(has_positive),
                "group_id": image_id,
            }
        )

        if i % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(index_csv, index=False)
            logger.info("checkpoint %d/%d DICOMs processed", i, len(dcm_candidates))

    if not rows:
        raise RuntimeError("SIIM preprocessing produced 0 samples")

    # Contract: dedup by group_id keep-last (no-op here — one group per image).
    df = pd.DataFrame(rows).drop_duplicates(subset=["group_id"], keep="last")

    # Fail-fast: every referenced file must exist on disk.
    img_dir = siim_root / "preprocessed" / "images"
    mask_dir = siim_root / "preprocessed" / "masks"
    valid = (
        df["image_path"].map(lambda p: (img_dir / Path(p).name).exists())
        & df["mask_path"].map(lambda p: (mask_dir / Path(p).name).exists())
    )
    if not valid.all():
        bad = df[~valid].iloc[0]
        raise RuntimeError(
            f"fail-fast: {int((~valid).sum())} rows reference missing files, "
            f"first: {bad['image_path']} / {bad['mask_path']}"
        )
    df = df.reset_index(drop=True)

    if len(df) < MIN_VALID_ROWS:
        raise RuntimeError(
            f"SIIM preprocessed index too small: {len(df)} < {MIN_VALID_ROWS} rows"
        )

    df.to_csv(index_csv, index=False)

    csv_ids = set(lookup)
    dcm_ids = {r["group_id"] for r in rows}
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary = {
        "index_csv": str(index_csv),
        "images": int(len(df)),
        "masks": int(len(df)),
        "positives": int(df["label_int"].sum()),
        "negatives": int((df["label_int"] == 0).sum()),
        "unique_group_id": int(df["group_id"].nunique()),
        "csv_rows_without_dicom": int(len(csv_ids - dcm_ids)),
        "dicoms_skipped_no_csv_row": int(len(skipped)),
        "skipped_ids": skipped,
        "files_written": int(n_written),
        "files_identical": int(n_identical),
        "files_resumed": int(n_resumed),
        "peak_rss_mb": float(peak_rss_mb),
    }
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--siim-root", type=Path, default=REPO_ROOT / "data" / "SIIM")
    p.add_argument("--dicom-dir", type=Path, default=REPO_ROOT / "data" / "SIIM" / "_dl" / "dicom-images-train")
    p.add_argument("--rle-csv", type=Path, default=None,
                   help="RLE CSV (default: auto-detect train-rle.csv / stage_2_train.csv next to --dicom-dir)")
    p.add_argument("--index-only", action="store_true",
                   help="Re-emit index.csv with patient-level group_id (PatientID -> "
                        "StudyInstanceUID fallback) from DICOM headers only; no pixel decode")
    args = p.parse_args()

    if args.index_only:
        summary = reindex_patient_groups(args.siim_root, args.dicom_dir)
        print("SIIM-ACR patient-level reindex summary")
        print(f"  index.csv            : {summary['index_csv']}")
        print(f"  images               : {summary['images']}")
        print(f"  group source         : {summary['group_source']}")
        print(f"  unique patients      : {summary['unique_groups']}")
        print(f"  group size min/med/max: {summary['group_size_min']} / "
              f"{summary['group_size_median']:.1f} / {summary['group_size_max']}")
        print(f"  patients with >1 image: {summary['patients_with_gt1_image']}")
        print(f"  fallback rows used   : {summary['fallback_rows_used']}")
        print(f"  provenance sidecar   : {summary['provenance_sidecar']}")
        print(f"  peak RSS             : {summary['peak_rss_mb']:.1f} MB")
        return 0

    summary = build(args.siim_root, args.dicom_dir, args.rle_csv)
    print("SIIM-ACR preprocessing summary")
    print(f"  index.csv            : {summary['index_csv']}")
    print(f"  images               : {summary['images']}")
    print(f"  masks                : {summary['masks']}")
    print(f"  positives            : {summary['positives']}")
    print(f"  negatives            : {summary['negatives']}")
    print(f"  unique group_id      : {summary['unique_group_id']} (post-dedup)")
    print(f"  csv rows w/o DICOM   : {summary['csv_rows_without_dicom']} (reported, not silent)")
    print(f"  DICOMs skipped (no CSV row): {summary['dicoms_skipped_no_csv_row']} (reported, not silent)")
    if summary["skipped_ids"]:
        for sid in summary["skipped_ids"]:
            print(f"    - {sid}")
    print(f"  files written        : {summary['files_written']}")
    print(f"  files byte-identical : {summary['files_identical']} (incl. pinned fig2 tiles)")
    print(f"  files resumed (skip) : {summary['files_resumed']}")
    print(f"  peak RSS             : {summary['peak_rss_mb']:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
