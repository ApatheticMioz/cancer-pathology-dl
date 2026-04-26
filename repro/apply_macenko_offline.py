from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
from PIL import Image

try:
    from repro.prepare import _build_pannuke_preprocessed
    from repro.utils import first_existing
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from repro.prepare import _build_pannuke_preprocessed
    from repro.utils import first_existing

_STAIN_MATRIX: np.ndarray | None = None
_MAX_CONC: np.ndarray | None = None


def _fit_macenko_reference(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image, 1.0, 255.0)
    od = -np.log(image / 255.0)
    flat = od.reshape(-1, 3)
    flat = flat[np.all(flat > 0.15, axis=1)]
    if flat.shape[0] < 32:
        raise RuntimeError("Macenko reference image has too few tissue pixels")

    cov = np.cov(flat, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order[:2]]

    projected = flat @ basis
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    low_angle, high_angle = np.percentile(angles, [1.0, 99.0])
    v1 = basis @ np.array([np.cos(low_angle), np.sin(low_angle)])
    v2 = basis @ np.array([np.cos(high_angle), np.sin(high_angle)])
    stains = np.stack([v1, v2], axis=1)
    stains = stains / np.linalg.norm(stains, axis=0, keepdims=True).clip(min=1e-8)

    concentrations = np.linalg.lstsq(stains, od.reshape(-1, 3).T, rcond=None)[0]
    max_conc = np.percentile(concentrations, 99.0, axis=1)
    return stains.astype(np.float32), max_conc.astype(np.float32)


def _init_worker(stain_matrix: np.ndarray, max_conc: np.ndarray) -> None:
    global _STAIN_MATRIX, _MAX_CONC
    _STAIN_MATRIX = stain_matrix
    _MAX_CONC = max_conc


def _macenko_apply(image: np.ndarray) -> np.ndarray:
    if _STAIN_MATRIX is None or _MAX_CONC is None:
        raise RuntimeError("Worker stain parameters were not initialized")

    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image, 1.0, 255.0)
    od = -np.log(image / 255.0)
    concentrations = np.linalg.lstsq(_STAIN_MATRIX, od.reshape(-1, 3).T, rcond=None)[0]
    concentrations = np.maximum(concentrations, 0.0)

    denom = np.percentile(concentrations, 99.0, axis=1).clip(min=1e-6)
    scale = _MAX_CONC / denom
    concentrations = concentrations * scale[:, None]

    recon = np.exp(-(_STAIN_MATRIX @ concentrations)).T.reshape(image.shape)
    return np.clip(recon * 255.0, 0.0, 255.0).astype(np.uint8)


def _process_one(src_path: str, dst_path: str) -> str:
    src = Path(src_path)
    dst = Path(dst_path)
    image = np.array(Image.open(src).convert("RGB"))
    norm = _macenko_apply(image)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(norm).save(dst)
    return str(dst)


def _select_reference(source_paths: list[Path]) -> np.ndarray:
    for path in source_paths:
        try:
            return np.array(Image.open(path).convert("RGB"))
        except Exception:
            continue
    raise RuntimeError("Could not load any source image for Macenko reference")


def _run_parallel(source_paths: list[Path], output_paths: list[Path], workers: int) -> None:
    pending = [(src, dst) for src, dst in zip(source_paths, output_paths) if not dst.exists()]
    if not pending:
        print("  all outputs already exist; skipping")
        return

    source_paths = [src for src, _ in pending]
    output_paths = [dst for _, dst in pending]

    ref = _select_reference(source_paths)
    stain_matrix, max_conc = _fit_macenko_reference(ref)

    total = len(source_paths)
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(stain_matrix, max_conc),
    ) as pool:
        futures = [
            pool.submit(_process_one, str(src), str(dst))
            for src, dst in zip(source_paths, output_paths)
        ]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 500 == 0 or done == total:
                print(f"  processed {done}/{total}")


def _prepare_panda_paths(base_dir: Path) -> tuple[list[Path], list[Path]]:
    panda_root = base_dir / "PANDA_raw"
    train_csv = panda_root / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing PANDA CSV: {train_csv}")

    image_dir = first_existing([panda_root / "train_images" / "train_images", panda_root / "train_images"])
    if image_dir is None:
        raise FileNotFoundError("Missing PANDA train_images directory")

    import pandas as pd

    df = pd.read_csv(train_csv)
    if "image_id" not in df.columns:
        raise ValueError("PANDA train.csv missing image_id")

    source_map: dict[str, Path] = {}
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
        for p in image_dir.rglob(ext):
            source_map[p.stem] = p

    out_dir = panda_root / "preprocessed_macenko" / "images"
    source_paths: list[Path] = []
    output_paths: list[Path] = []
    for image_id in df["image_id"].astype(str):
        src = source_map.get(image_id)
        if src is None:
            continue
        source_paths.append(src)
        output_paths.append(out_dir / f"{image_id}.png")

    if not source_paths:
        raise RuntimeError("No PANDA source images matched train.csv")

    return source_paths, output_paths


def _prepare_pannuke_paths(base_dir: Path) -> tuple[list[Path], list[Path]]:
    pannuke_root = base_dir / "pannuke"
    index_csv = pannuke_root / "preprocessed" / "index.csv"
    if not index_csv.exists():
        print("  PanNuke preprocessed index missing; building base preprocessed assets first")
        _build_pannuke_preprocessed(pannuke_root)

    image_dir = pannuke_root / "preprocessed" / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing PanNuke source image directory: {image_dir}")

    source_paths = sorted([p for p in image_dir.glob("*.png") if p.is_file()])
    if not source_paths:
        raise RuntimeError("No PanNuke preprocessed source PNGs found")

    out_dir = pannuke_root / "preprocessed_macenko" / "images"
    output_paths = [out_dir / p.name for p in source_paths]
    return source_paths, output_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply offline Macenko normalization for PANDA/PanNuke")
    parser.add_argument("--datasets", nargs="+", default=["panda", "pannuke"], choices=["panda", "pannuke"])
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent

    if "panda" in args.datasets:
        print("[PANDA] collecting source images")
        panda_src, panda_dst = _prepare_panda_paths(base_dir)
        print(f"[PANDA] running Macenko offline on {len(panda_src)} images with workers={args.workers}")
        _run_parallel(panda_src, panda_dst, workers=args.workers)

    if "pannuke" in args.datasets:
        print("[PANNUKE] collecting source images")
        pannuke_src, pannuke_dst = _prepare_pannuke_paths(base_dir)
        print(f"[PANNUKE] running Macenko offline on {len(pannuke_src)} images with workers={args.workers}")
        _run_parallel(pannuke_src, pannuke_dst, workers=args.workers)

    print("Offline Macenko preprocessing complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
