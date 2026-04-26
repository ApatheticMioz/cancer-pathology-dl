from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

try:
    from repro.config import (
        CHECKPOINT_DIR,
        DATASET_AUDIT_FILE,
        DATASET_ROOTS,
        ISIC_KAGGLE_FALLBACK_REFS,
        ISIC_URLS,
        PANDA_COMPETITION,
        SIIM_COMPETITION,
    )
    from repro.utils import atomic_json_write, count_files, first_existing, now_iso
except ImportError:
    from .config import (
        CHECKPOINT_DIR,
        DATASET_AUDIT_FILE,
        DATASET_ROOTS,
        ISIC_KAGGLE_FALLBACK_REFS,
        ISIC_URLS,
        PANDA_COMPETITION,
        SIIM_COMPETITION,
    )
    from .utils import atomic_json_write, count_files, first_existing, now_iso

try:
    import pydicom

    PYDICOM_AVAILABLE = True
except ImportError:
    pydicom = None
    PYDICOM_AVAILABLE = False


def _get_kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)


def _purge_children(root: Path) -> None:
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return
    for child in root.iterdir():
        _safe_remove(child)


def _is_valid_zip(path: Path) -> bool:
    return path.exists() and zipfile.is_zipfile(path)


def _extract_zip(path: Path, target_dir: Path, delete_after: bool = True) -> None:
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(target_dir)
    if delete_after:
        path.unlink(missing_ok=True)


def _extract_all_zips(root: Path, delete_after: bool = True) -> int:
    extracted = 0
    seen: set[Path] = set()

    while True:
        zips = sorted([p for p in root.rglob("*.zip") if p.is_file()])
        pending = [p for p in zips if p.resolve() not in seen]
        if not pending:
            break

        changed = False
        for zf in pending:
            seen.add(zf.resolve())
            if not _is_valid_zip(zf):
                continue
            _extract_zip(zf, zf.parent, delete_after=delete_after)
            extracted += 1
            changed = True

        if not changed:
            break
    return extracted


def _zip_entry_count(path: Path, suffixes: tuple[str, ...] | None = None) -> int:
    if not _is_valid_zip(path):
        return 0

    normalized_suffixes = tuple(s.lower() for s in suffixes) if suffixes else None

    try:
        with zipfile.ZipFile(path, "r") as zf:
            count = 0
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if normalized_suffixes and not info.filename.lower().endswith(normalized_suffixes):
                    continue
                count += 1
            return count
    except Exception:
        return 0


def _remove_download_artifacts(path: Path) -> None:
    path.unlink(missing_ok=True)
    path.with_suffix(path.suffix + ".part").unlink(missing_ok=True)


def _download_file_via_curl(url: str, out_path: Path, retries: int = 6) -> None:
    curl_bin = shutil.which("curl.exe") or shutil.which("curl")
    if not curl_bin:
        raise RuntimeError("curl is not available on PATH")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    # If a valid final file already exists, keep it.
    if _is_valid_zip(out_path):
        return

    # If the final path contains a larger invalid partial than .part,
    # carry that progress forward by resuming from the larger file.
    if out_path.exists() and not _is_valid_zip(out_path):
        out_size = out_path.stat().st_size
        tmp_size = tmp.stat().st_size if tmp.exists() else -1
        if out_size > tmp_size:
            out_path.replace(tmp)

    # If a previous run already completed into .part, promote it.
    if _is_valid_zip(tmp):
        tmp.replace(out_path)
        return

    last_error = None
    for attempt in range(1, retries + 1):
        cmd = [
            curl_bin,
            "-L",
            "--fail",
            "--http1.1",
            "--connect-timeout",
            "30",
            "--output",
            str(tmp),
        ]

        if tmp.exists() and tmp.stat().st_size > 0:
            cmd.extend(["-C", "-"])

        cmd.append(url)

        try:
            subprocess.run(cmd, check=True)
            if _is_valid_zip(tmp):
                tmp.replace(out_path)
                return
            last_error = RuntimeError(f"downloaded file is not a valid zip yet: {tmp}")
        except Exception as ex:
            last_error = ex

        if attempt < retries:
            time.sleep(min(20, attempt * 3))

    raise RuntimeError(f"curl download failed for {url}: {last_error}")


def _download_file(url: str, out_path: Path, retries: int = 3) -> None:
    # Prefer curl in this environment (Python SSL EOFs were observed on ISIC S3).
    try:
        _download_file_via_curl(url, out_path, retries=max(4, retries * 2))
        return
    except Exception:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None

    for attempt in range(1, retries + 1):
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(out_path)
            return
        except Exception as ex:
            last_error = ex
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(15, attempt * 3))

    raise RuntimeError(f"Failed to download {url}: {last_error}")


def _download_kaggle_competition_zip(competition: str, root: Path, force: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / f"{competition}.zip"
    download_root = root

    if zip_path.exists() and _is_valid_zip(zip_path) and not force:
        return zip_path

    if zip_path.exists() and (force or not _is_valid_zip(zip_path)):
        try:
            zip_path.unlink(missing_ok=True)
        except PermissionError:
            # If the old archive is locked by another process, download to a temp dir.
            download_root = root / "_kaggle_download_tmp" / competition
            download_root.mkdir(parents=True, exist_ok=True)

    download_zip_path = download_root / f"{competition}.zip"
    if download_zip_path.exists() and not _is_valid_zip(download_zip_path):
        download_zip_path.unlink(missing_ok=True)

    api = _get_kaggle_api()
    last_error = None
    for attempt in range(1, 6):
        try:
            api.competition_download_files(competition, path=str(download_root), force=True, quiet=False)
            last_error = None
            break
        except Exception as ex:
            last_error = ex
            if attempt < 5:
                time.sleep(min(30, attempt * 4))

    if last_error is not None:
        raise RuntimeError(f"Kaggle competition download failed for {competition}: {last_error}")

    if not _is_valid_zip(download_zip_path):
        raise RuntimeError(f"Downloaded Kaggle archive is invalid: {download_zip_path}")
    return download_zip_path


def _download_kaggle_dataset_zip(dataset_ref: str, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    api = _get_kaggle_api()

    before = {p.resolve() for p in root.glob("*.zip")}
    last_error = None
    for attempt in range(1, 6):
        try:
            api.dataset_download_files(dataset_ref, path=str(root), force=True, quiet=False, unzip=False)
            last_error = None
            break
        except Exception as ex:
            last_error = ex
            if attempt < 5:
                time.sleep(min(30, attempt * 4))

    if last_error is not None:
        raise RuntimeError(f"Kaggle dataset download failed for {dataset_ref}: {last_error}")

    after = [p for p in root.glob("*.zip") if p.resolve() not in before]

    if not after:
        # Fallback for slug-based naming.
        slug = dataset_ref.split("/")[-1]
        candidate = root / f"{slug}.zip"
        if candidate.exists():
            after = [candidate]

    if not after:
        raise RuntimeError(f"Kaggle dataset download produced no archive for {dataset_ref}")

    zip_path = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    if not _is_valid_zip(zip_path):
        raise RuntimeError(f"Downloaded Kaggle dataset archive invalid: {zip_path}")
    return zip_path


def _find_tcga_pairs(root: Path) -> tuple[int, int, int]:
    image_count = 0
    mask_count = 0
    pair_count = 0
    for patient in [d for d in root.iterdir() if d.is_dir()]:
        for img in patient.glob("*.tif"):
            if img.name.endswith("_mask.tif"):
                mask_count += 1
                continue
            image_count += 1
            mask = img.with_name(img.stem + "_mask.tif")
            if mask.exists():
                pair_count += 1
    return image_count, mask_count, pair_count


def _isic_layout(root: Path) -> dict:
    label_csv = first_existing(
        [
            root / "labels.csv",
            root / "ISIC2018_Task3_Training_GroundTruth.csv",
            root
            / "ISIC2018_Task3_Training_GroundTruth"
            / "ISIC2018_Task3_Training_GroundTruth.csv",
            root
            / "ISIC2018_Task3_Training_GroundTruth"
            / "ISIC2018_Task3_Training_GroundTruth.txt",
        ]
    )

    image_dir = first_existing(
        [
            root / "images",
            root / "ISIC2018_Task1-2_Training_Input",
            root / "ISIC2018_Task3_Training_Input",
        ]
    )

    mask_dir = first_existing(
        [
            root / "masks",
            root / "ISIC2018_Task1_Training_GroundTruth",
        ]
    )

    image_count = count_files(image_dir, ["*.jpg", "*.jpeg", "*.png"]) if image_dir else 0
    mask_count = count_files(mask_dir, ["*.png", "*.jpg", "*.jpeg"]) if mask_dir else 0

    return {
        "label_csv": str(label_csv) if label_csv else None,
        "image_dir": str(image_dir) if image_dir else None,
        "mask_dir": str(mask_dir) if mask_dir else None,
        "image_count": int(image_count),
        "mask_count": int(mask_count),
        "ready": bool(label_csv and image_dir and mask_dir and image_count >= 2000 and mask_count >= 2000),
    }


def _panda_layout(root: Path) -> dict:
    train_csv = root / "train.csv"
    image_dir = first_existing([root / "train_images" / "train_images", root / "train_images"])
    mask_dir = first_existing([root / "train_label_masks" / "train_label_masks", root / "train_label_masks"])

    image_count = 0
    mask_count = 0
    if image_dir:
        image_count = count_files(image_dir, ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"])
    if mask_dir:
        mask_count = count_files(mask_dir, ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"])

    return {
        "train_csv": str(train_csv) if train_csv.exists() else None,
        "image_dir": str(image_dir) if image_dir else None,
        "mask_dir": str(mask_dir) if mask_dir else None,
        "image_count": int(image_count),
        "mask_count": int(mask_count),
        "ready": bool(train_csv.exists() and image_dir and mask_dir and image_count >= 10000 and mask_count >= 10000),
    }


def _siim_layout(root: Path) -> dict:
    pre_dir = root / "preprocessed"
    index_csv = pre_dir / "index.csv"
    image_dir = pre_dir / "images"
    mask_dir = pre_dir / "masks"

    # Check multiple possible DICOM locations
    dcm_pneumothorax = count_files(root / "pneumothorax" / "dicom-images-train", ["*.dcm"])
    dcm_train = count_files(root / "dicom-images-train", ["*.dcm"])
    dcm_stage2 = count_files(root / "stage_2_images", ["*.dcm"])
    dcm_total = max(dcm_pneumothorax, dcm_train, dcm_stage2)

    image_count = count_files(image_dir, ["*.png"])
    mask_count = count_files(mask_dir, ["*.png"])

    row_count = 0
    if index_csv.exists():
        try:
            row_count = int(sum(1 for _ in index_csv.open("r", encoding="utf-8")) - 1)
        except Exception:
            row_count = 0

    return {
        "index_csv": str(index_csv) if index_csv.exists() else None,
        "image_dir": str(image_dir) if image_dir.exists() else None,
        "mask_dir": str(mask_dir) if mask_dir.exists() else None,
        "dcm_count": int(dcm_total),
        "image_count": int(image_count),
        "mask_count": int(mask_count),
        "index_rows": int(max(0, row_count)),
        "ready": bool(index_csv.exists() and image_count >= 10000 and mask_count >= 10000 and row_count >= 10000),
    }


def _decode_siim_rle(encoded_pixels: str, height: int, width: int) -> np.ndarray:
    mask = np.zeros(height * width, dtype=np.uint8)
    if not encoded_pixels or encoded_pixels == "-1" or encoded_pixels.lower() == "nan":
        return mask.reshape((height, width), order="F")

    values = [int(v) for v in encoded_pixels.split()]
    starts = np.asarray(values[0::2], dtype=np.int64) - 1
    lengths = np.asarray(values[1::2], dtype=np.int64)
    ends = starts + lengths

    for start, end in zip(starts, ends):
        mask[start:end] = 255

    return mask.reshape((height, width), order="F")


def _normalize_dicom_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    out = ((arr - lo) / (hi - lo) * 255.0).clip(0, 255)
    return out.astype(np.uint8)


def _siim_rle_lookup(root: Path) -> dict[str, list[str]]:
    csv_path = first_existing([root / "train-rle.csv", root / "stage_2_train.csv"])
    if csv_path is None:
        raise FileNotFoundError("SIIM train-rle.csv/stage_2_train.csv missing")

    df = pd.read_csv(csv_path)
    # Strip whitespace from column names (some downloads have " EncodedPixels" with leading space)
    df.columns = df.columns.str.strip()
    if "ImageId" not in df.columns or "EncodedPixels" not in df.columns:
        raise ValueError(f"Invalid SIIM CSV columns in {csv_path}")

    lookup: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        image_id = str(row["ImageId"]).strip()
        encoded = str(row["EncodedPixels"]).strip()
        lookup.setdefault(image_id, []).append(encoded)
    return lookup


def _build_siim_preprocessed(root: Path) -> dict:
    if not PYDICOM_AVAILABLE:
        raise RuntimeError("pydicom is required for SIIM preprocessing")

    lookup = _siim_rle_lookup(root)
    if not lookup:
        raise RuntimeError("SIIM RLE lookup is empty")

    pre_dir = root / "preprocessed"
    image_dir = pre_dir / "images"
    mask_dir = pre_dir / "masks"
    index_csv = pre_dir / "index.csv"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Try multiple possible DICOM locations
    dcm_candidates = sorted((root / "pneumothorax" / "dicom-images-train").rglob("*.dcm"))
    if len(dcm_candidates) < 5000:
        dcm_candidates = sorted((root / "dicom-images-train").rglob("*.dcm"))
    if len(dcm_candidates) < 5000:
        dcm_candidates = sorted((root / "stage_2_images").rglob("*.dcm"))

    if not dcm_candidates:
        raise RuntimeError("No SIIM DICOM files found for preprocessing")

    rows = []
    checkpoint_every = 500

    for i, dcm_path in enumerate(dcm_candidates, start=1):
        image_id = dcm_path.stem
        rles = lookup.get(image_id)

        if rles is None:
            try:
                header = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                sop_uid = str(getattr(header, "SOPInstanceUID", "")).strip()
                if sop_uid:
                    image_id = sop_uid
                    rles = lookup.get(sop_uid)
            except Exception:
                rles = None

        if rles is None:
            continue

        safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", image_id)
        image_out = image_dir / f"{safe_id}.png"
        mask_out = mask_dir / f"{safe_id}_mask.png"
        has_positive = any(r and r != "-1" and r.lower() != "nan" for r in rles)

        if not image_out.exists() or not mask_out.exists():
            ds = pydicom.dcmread(dcm_path)
            px = _normalize_dicom_to_uint8(ds.pixel_array)
            h, w = px.shape
            image_rgb = np.stack([px, px, px], axis=-1)
            Image.fromarray(image_rgb).save(image_out)

            mask = np.zeros((h, w), dtype=np.uint8)
            if has_positive:
                for encoded in rles:
                    if encoded and encoded != "-1" and encoded.lower() != "nan":
                        mask = np.maximum(mask, _decode_siim_rle(encoded, h, w))
            Image.fromarray(mask).save(mask_out)

        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_out),
                "mask_path": str(mask_out),
                "label_int": int(has_positive),
                "group_id": image_id,
                "height": -1,
                "width": -1,
            }
        )

        if i % checkpoint_every == 0 and rows:
            pd.DataFrame(rows).drop_duplicates(subset=["image_id"], keep="last").to_csv(index_csv, index=False)

    if not rows:
        raise RuntimeError("SIIM preprocessing produced 0 samples")

    df = pd.DataFrame(rows).drop_duplicates(subset=["image_id"], keep="last")
    valid = df["image_path"].map(os.path.exists) & df["mask_path"].map(os.path.exists)
    df = df[valid].reset_index(drop=True)
    df.to_csv(index_csv, index=False)

    return {
        "index_csv": str(index_csv),
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "image_count": int(len(df)),
        "mask_count": int(len(df)),
        "index_rows": int(len(df)),
        "ready": len(df) >= 10000,
    }


def prepare_tcga(root: Path, force_redownload: bool = False) -> dict:
    if force_redownload:
        raise RuntimeError("TCGA force redownload is not supported automatically. Place TCGA data in final/TCGA.")

    if not root.exists():
        raise FileNotFoundError(f"TCGA root missing: {root}")

    image_count, mask_count, pair_count = _find_tcga_pairs(root)
    ready = pair_count >= 3000
    if not ready:
        raise RuntimeError(f"TCGA incomplete: images={image_count}, masks={mask_count}, pairs={pair_count}")

    print(f"  [TCGA] ready with {pair_count} image/mask pairs")

    return {
        "root": str(root),
        "image_count": image_count,
        "mask_count": mask_count,
        "pair_count": pair_count,
        "ready": True,
    }


def prepare_isic(root: Path, force_redownload: bool = False) -> dict:
    state = _isic_layout(root)
    if state["ready"] and not force_redownload:
        print(
            f"  [ISIC] ready with images={state['image_count']} masks={state['mask_count']} "
            f"labels={state['label_csv']}"
        )
        return state

    root.mkdir(parents=True, exist_ok=True)

    extracted_local = _extract_all_zips(root, delete_after=False)
    if extracted_local > 0:
        print(f"  [ISIC] extracted {extracted_local} local archive(s)")

    state = _isic_layout(root)
    if state["ready"] and not force_redownload:
        print(
            f"  [ISIC] ready after local recovery: images={state['image_count']} masks={state['mask_count']} "
            f"labels={state['label_csv']}"
        )
        return state

    archive_specs = [
        {
            "filename": "ISIC2018_Task1-2_Training_Input.zip",
            "url": ISIC_URLS["ISIC2018_Task1-2_Training_Input.zip"],
            "suffixes": (".jpg", ".jpeg", ".png"),
            "min_entries": 2000,
            "extract_dir": "ISIC2018_Task1-2_Training_Input",
        },
        {
            "filename": "ISIC2018_Task1_Training_GroundTruth.zip",
            "url": ISIC_URLS["ISIC2018_Task1_Training_GroundTruth.zip"],
            "suffixes": (".png", ".jpg", ".jpeg"),
            "min_entries": 2000,
            "extract_dir": "ISIC2018_Task1_Training_GroundTruth",
        },
        {
            "filename": "ISIC2018_Task3_Training_GroundTruth.zip",
            "url": ISIC_URLS["ISIC2018_Task3_Training_GroundTruth.zip"],
            "suffixes": (".csv", ".txt"),
            "min_entries": 1,
            "extract_dir": "ISIC2018_Task3_Training_GroundTruth",
        },
    ]

    def _needs_component(filename: str, current_state: dict) -> bool:
        if filename == "ISIC2018_Task1-2_Training_Input.zip":
            return current_state["image_count"] < 2000
        if filename == "ISIC2018_Task1_Training_GroundTruth.zip":
            return current_state["mask_count"] < 2000
        if filename == "ISIC2018_Task3_Training_GroundTruth.zip":
            return not bool(current_state["label_csv"])
        return True

    def _download_and_extract_component(spec: dict, force_clean: bool) -> tuple[bool, str | None]:
        out_path = root / spec["filename"]
        suffixes = spec["suffixes"]
        min_entries = int(spec["min_entries"])
        extract_dir = root / spec["extract_dir"]
        extract_errors: list[str] = []

        if force_clean:
            _remove_download_artifacts(out_path)
            _safe_remove(extract_dir)

        for attempt in range(2):
            if attempt > 0:
                # A failed extract can indicate a CRC-corrupt resumed archive.
                # Retry once with a clean download and empty extracted component dir.
                _remove_download_artifacts(out_path)
                _safe_remove(extract_dir)

            entry_count = _zip_entry_count(out_path, suffixes)
            if not _is_valid_zip(out_path) or entry_count < min_entries:
                # Try a resumed/normal download first.
                try:
                    _download_file(spec["url"], out_path, retries=3)
                except Exception as ex:
                    return False, str(ex)

                entry_count = _zip_entry_count(out_path, suffixes)

            if entry_count < min_entries:
                return (
                    False,
                    f"archive content check failed for {spec['filename']}: "
                    f"entry_count={entry_count} expected>={min_entries}",
                )

            try:
                _extract_zip(out_path, root, delete_after=True)
                return True, None
            except Exception as ex:
                extract_errors.append(str(ex))
                err_text = str(ex).lower()
                recoverable = (
                    "crc" in err_text
                    or "badzip" in err_text
                    or "file is not a zip file" in err_text
                    or "truncated" in err_text
                    or "unexpected end of data" in err_text
                )
                if attempt == 0 and recoverable:
                    continue
                _remove_download_artifacts(out_path)
                return False, f"extract failed for {spec['filename']}: {ex}"

        if extract_errors:
            return False, f"extract failed for {spec['filename']}: {extract_errors[-1]}"
        return False, f"extract failed for {spec['filename']}: unknown error"

    official_errors = {}

    print("  [ISIC] downloading official 2018 archives")
    for spec in archive_specs:
        state = _isic_layout(root)
        if not force_redownload and not _needs_component(spec["filename"], state):
            continue

        ok, err = _download_and_extract_component(spec, force_clean=force_redownload)
        if not ok and err is not None:
            official_errors[spec["filename"]] = err

    state = _isic_layout(root)
    if state["ready"]:
        print(f"  [ISIC] ready using official archives")
        return state

    # Fall back to Kaggle mirrors if direct S3 fetch fails in the current network.
    print("  [ISIC] official download did not complete dataset, trying Kaggle mirrors")
    for ref in ISIC_KAGGLE_FALLBACK_REFS:
        try:
            print(f"  [ISIC] trying Kaggle dataset: {ref}")
            zip_path = _download_kaggle_dataset_zip(ref, root)
            _extract_zip(zip_path, root, delete_after=True)
            _extract_all_zips(root)
            state = _isic_layout(root)
            if state["ready"]:
                print(f"  [ISIC] ready using Kaggle mirror: {ref}")
                return state
        except Exception as ex:
            print(f"  [ISIC] mirror failed ({ref}): {ex}")
            continue

    state = _isic_layout(root)
    if not state["ready"]:
        msg = (
            "ISIC dataset still incomplete after recovery. "
            f"images={state['image_count']} masks={state['mask_count']} labels={state['label_csv']}"
        )
        if official_errors:
            msg += f"; official_errors={official_errors}"
        raise RuntimeError(msg)
    return state


def prepare_panda(root: Path, force_redownload: bool = False) -> dict:
    state = _panda_layout(root)
    if state["ready"] and not force_redownload:
        print(
            f"  [PANDA] ready with images={state['image_count']} masks={state['mask_count']}"
        )
        return state

    root.mkdir(parents=True, exist_ok=True)

    extracted_local = _extract_all_zips(root, delete_after=False)
    if extracted_local > 0:
        print(f"  [PANDA] extracted {extracted_local} local archive(s)")

    state = _panda_layout(root)
    if state["ready"] and not force_redownload:
        print(
            f"  [PANDA] ready after local recovery with images={state['image_count']} masks={state['mask_count']}"
        )
        return state

    train_csv_path = root / "train.csv"
    backup_bytes = train_csv_path.read_bytes() if train_csv_path.exists() else None

    comp_zip = root / f"{PANDA_COMPETITION}.zip"
    needs_download = force_redownload or not _is_valid_zip(comp_zip)
    if needs_download:
        print(f"  [PANDA] downloading from Kaggle competition: {PANDA_COMPETITION}")
        comp_zip = _download_kaggle_competition_zip(PANDA_COMPETITION, root, force=True)
    else:
        print("  [PANDA] using existing competition archive")

    if not _is_valid_zip(comp_zip):
        raise RuntimeError(f"PANDA competition archive is invalid: {comp_zip}")

    _extract_zip(comp_zip, root, delete_after=True)
    _extract_all_zips(root)

    if backup_bytes is not None and not train_csv_path.exists():
        train_csv_path.write_bytes(backup_bytes)

    state = _panda_layout(root)
    if not state["ready"]:
        raise RuntimeError(
            f"PANDA dataset incomplete after redownload/extraction: images={state['image_count']} masks={state['mask_count']}"
        )
    return state


def prepare_siim(root: Path, force_redownload: bool = False) -> dict:
    state = _siim_layout(root)
    if state["ready"] and not force_redownload:
        print(
            f"  [SIIM] ready with rows={state['index_rows']} images={state['image_count']} masks={state['mask_count']}"
        )
        return state

    root.mkdir(parents=True, exist_ok=True)

    needs_payload = force_redownload or state["dcm_count"] < 10000
    if needs_payload:
        for p in [
            root / "dicom-images-train",
            root / "dicom-images-test",
            root / "stage_2_images",
            root / "preprocessed",
        ]:
            _safe_remove(p)

        for zf in root.glob("*.zip"):
            zf.unlink(missing_ok=True)

        print(f"  [SIIM] downloading from Kaggle competition: {SIIM_COMPETITION}")
        zip_path = _download_kaggle_competition_zip(SIIM_COMPETITION, root)
        _extract_zip(zip_path, root, delete_after=True)
        _extract_all_zips(root)

    print("  [SIIM] building preprocessed PNG + mask index")
    built = _build_siim_preprocessed(root)
    if not built["ready"]:
        raise RuntimeError(
            f"SIIM preprocessing incomplete after rebuild: rows={built['index_rows']}"
        )

    return {**_siim_layout(root), **built}


def prepare_datasets(
    datasets: list[str],
    force_redownload_all: bool = False,
) -> dict:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": now_iso(),
        "datasets": {},
    }

    for dataset in datasets:
        root = DATASET_ROOTS[dataset]
        if dataset == "tcga":
            results["datasets"][dataset] = prepare_tcga(root, force_redownload=force_redownload_all)
        elif dataset == "isic":
            results["datasets"][dataset] = prepare_isic(root, force_redownload=force_redownload_all)
        elif dataset == "panda":
            results["datasets"][dataset] = prepare_panda(root, force_redownload=force_redownload_all)
        elif dataset == "siim":
            results["datasets"][dataset] = prepare_siim(root, force_redownload=force_redownload_all)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    atomic_json_write(DATASET_AUDIT_FILE, results)
    return results
