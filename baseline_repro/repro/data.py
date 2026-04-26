from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset

try:
    from repro.config import DATASET_META
    from repro.utils import first_existing
except ImportError:
    from .config import DATASET_META
    from .utils import first_existing


def build_transforms(img_size: int):
    train_tf = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Affine(shear=(-10, 10), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    val_tf = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    return train_tf, val_tf


def make_group_split(labels: np.ndarray, groups: np.ndarray, seed: int, test_size: float = 0.2):
    # If groups are effectively unique per sample (e.g., PANDA image_id),
    # use stratified splitting to preserve class balance in train/val.
    if len(np.unique(groups)) == len(labels):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        splits = list(splitter.split(np.zeros(len(labels)), labels))
        if not splits:
            raise RuntimeError("Could not create stratified split")
        return splits[0]

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    splits = list(splitter.split(np.zeros(len(labels)), labels, groups))
    if not splits:
        raise RuntimeError("Could not create group split")
    return splits[0]


def _resolve_isic_image_path(image_root: Path, image_name: str) -> Path | None:
    candidate = image_root / image_name
    if candidate.exists():
        return candidate

    stem = Path(image_name).stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        c = image_root / f"{stem}{ext}"
        if c.exists():
            return c
    return None


def _resolve_isic_mask_path(mask_root: Path, stem: str) -> Path | None:
    for name in [
        f"{stem}_segmentation.png",
        f"{stem}_segmentation.jpg",
        f"{stem}_mask.png",
        f"{stem}_mask.jpg",
        f"{stem}.png",
    ]:
        c = mask_root / name
        if c.exists():
            return c
    return None


def _normalize_isic_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"image_filename", "label_int"}.issubset(out.columns):
        out["group_id"] = out.get("patient_id", out["image_filename"])
        out["label_int"] = out["label_int"].astype(int)
        return out[["image_filename", "label_int", "group_id"]]

    one_hot_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    if all(col in out.columns for col in one_hot_cols):
        image_col = None
        for c in ["image", "image_id", "image_name", out.columns[0]]:
            if c in out.columns:
                image_col = c
                break
        if image_col is None:
            raise ValueError("Could not infer ISIC image column.")

        names = out[image_col].astype(str)
        names = names.where(names.str.contains(r"\.\w+$", regex=True), names + ".jpg")

        out2 = pd.DataFrame()
        out2["image_filename"] = names
        out2["label_int"] = out[one_hot_cols].values.argmax(axis=1).astype(int)
        out2["group_id"] = out.get("lesion_id", out2["image_filename"])
        return out2[["image_filename", "label_int", "group_id"]]

    if {"image", "dx"}.issubset(out.columns):
        mapping = {
            "mel": 0,
            "nv": 1,
            "bcc": 2,
            "akiec": 3,
            "bkl": 4,
            "df": 5,
            "vasc": 6,
        }

        out2 = pd.DataFrame()
        out2["image_filename"] = out["image"].astype(str)
        out2["image_filename"] = out2["image_filename"].where(
            out2["image_filename"].str.contains(r"\.\w+$", regex=True),
            out2["image_filename"] + ".jpg",
        )
        out2["label_int"] = out["dx"].astype(str).str.lower().map(mapping)
        out2 = out2.dropna(subset=["label_int"])
        out2["label_int"] = out2["label_int"].astype(int)
        out2["group_id"] = out.get("lesion_id", out2["image_filename"])
        return out2[["image_filename", "label_int", "group_id"]]

    raise ValueError("ISIC labels CSV format not recognized")


def parse_tcga(root: Path) -> dict:
    images = []
    masks = []
    labels = []
    groups = []

    for patient_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        for img in sorted(patient_dir.glob("*.tif")):
            if img.name.endswith("_mask.tif"):
                continue
            mask = img.with_name(img.stem + "_mask.tif")
            if not mask.exists():
                continue

            mask_np = np.array(Image.open(mask).convert("L"))
            label = int(np.any(mask_np > 0))

            images.append(str(img))
            masks.append(str(mask))
            labels.append(label)
            groups.append(patient_dir.name)

    if not images:
        raise RuntimeError("No TCGA image/mask pairs found")

    return {
        "images": np.array(images),
        "masks": np.array(masks),
        "labels": np.array(labels, dtype=np.int64),
        "groups": np.array(groups),
    }


def parse_isic(root: Path) -> dict:
    label_csv = first_existing(
        [
            root / "labels.csv",
            root / "ISIC2018_Task3_Training_GroundTruth.csv",
            root / "ISIC2018_Task3_Training_GroundTruth" / "ISIC2018_Task3_Training_GroundTruth.csv",
            root / "ISIC2018_Task3_Training_GroundTruth" / "ISIC2018_Task3_Training_GroundTruth.txt",
        ]
    )
    if label_csv is None:
        raise FileNotFoundError("ISIC labels CSV not found")

    image_root = first_existing(
        [
            root / "images",
            root / "ISIC2018_Task1-2_Training_Input",
            root / "ISIC2018_Task3_Training_Input",
            root,
        ]
    )
    if image_root is None:
        raise FileNotFoundError("ISIC image directory not found")

    mask_root = first_existing([root / "masks", root / "ISIC2018_Task1_Training_GroundTruth"])
    if mask_root is None:
        raise FileNotFoundError("ISIC mask directory not found")

    labels_df = pd.read_csv(label_csv)
    labels_df = _normalize_isic_labels(labels_df)

    # ISIC archives can introduce nested folders; build recursive indexes by stem.
    image_map = _index_stems(image_root, ["*.jpg", "*.jpeg", "*.png", "*.bmp"])
    raw_mask_map = _index_stems(mask_root, ["*.png", "*.jpg", "*.jpeg"])

    mask_lookup = {}
    for mask_stem, mask_path in raw_mask_map.items():
        mask_lookup.setdefault(mask_stem, mask_path)
        if mask_stem.endswith("_segmentation"):
            mask_lookup.setdefault(mask_stem[: -len("_segmentation")], mask_path)
        if mask_stem.endswith("_mask"):
            mask_lookup.setdefault(mask_stem[: -len("_mask")], mask_path)

    images = []
    masks = []
    labels = []
    groups = []

    for _, row in labels_df.iterrows():
        img_name = str(row["image_filename"])
        stem = Path(img_name).stem

        image_path = image_map.get(stem)
        if image_path is None:
            direct_img = _resolve_isic_image_path(image_root, img_name)
            image_path = str(direct_img) if direct_img else None

        mask_path = mask_lookup.get(stem)
        if mask_path is None:
            direct_mask = _resolve_isic_mask_path(mask_root, stem)
            mask_path = str(direct_mask) if direct_mask else None

        if image_path is None or mask_path is None:
            continue

        images.append(image_path)
        masks.append(mask_path)
        labels.append(int(row["label_int"]))
        groups.append(str(row["group_id"]))

    if not images:
        # Some mirrors provide Task1-2 segmentation assets with Task3 labels
        # whose image IDs do not overlap. Fall back to segmentation pairs so
        # the dataset remains usable.
        shared_stems = sorted(set(image_map.keys()) & set(mask_lookup.keys()))
        for stem in shared_stems:
            images.append(image_map[stem])
            masks.append(mask_lookup[stem])
            labels.append(0)
            groups.append(stem)

    if not images:
        raise RuntimeError("No ISIC image/mask pairs available")

    return {
        "images": np.array(images),
        "masks": np.array(masks),
        "labels": np.array(labels, dtype=np.int64),
        "groups": np.array(groups),
    }


def _index_stems(directory: Path, patterns: list[str]) -> dict[str, str]:
    mapping = {}
    for pattern in patterns:
        for p in directory.rglob(pattern):
            if p.is_file():
                mapping[p.stem] = str(p)
    return mapping


def parse_panda(root: Path) -> dict:
    train_csv = root / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError("PANDA train.csv missing")

    image_dir = first_existing([root / "train_images" / "train_images", root / "train_images"])
    mask_dir = first_existing([root / "train_label_masks" / "train_label_masks", root / "train_label_masks"])
    if image_dir is None or mask_dir is None:
        raise FileNotFoundError("PANDA train_images/train_label_masks directory missing")

    image_map = _index_stems(image_dir, ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"])
    mask_map = _index_stems(mask_dir, ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"])
    mask_lookup = {
        (stem[:-5] if stem.endswith("_mask") else stem): path
        for stem, path in mask_map.items()
    }

    df = pd.read_csv(train_csv)
    if "image_id" not in df.columns or "isup_grade" not in df.columns:
        raise ValueError("PANDA train.csv missing image_id/isup_grade")

    images = []
    masks = []
    labels = []
    groups = []

    for _, row in df.iterrows():
        image_id = str(row["image_id"]).strip()
        image_path = image_map.get(image_id)

        mask_path = mask_lookup.get(image_id)

        if image_path is None or mask_path is None:
            continue

        try:
            isup_grade = int(row["isup_grade"])
        except Exception:
            continue

        if isup_grade < 0 or isup_grade > 5:
            continue

        images.append(image_path)
        masks.append(mask_path)
        labels.append(isup_grade)
        groups.append(image_id)

    if not images:
        raise RuntimeError("No PANDA image/mask pairs matched train.csv")

    return {
        "images": np.array(images),
        "masks": np.array(masks),
        "labels": np.array(labels, dtype=np.int64),
        "groups": np.array(groups),
    }


def parse_pannuke(root: Path) -> dict:
    index_csv = first_existing([root / "preprocessed" / "index.csv", root / "index.csv"])
    if index_csv is None:
        raise FileNotFoundError("PanNuke preprocessed/index.csv missing")

    df = pd.read_csv(index_csv)
    required = {"image_path", "mask_path", "label_int", "group_id"}
    if not required.issubset(df.columns):
        raise ValueError("PanNuke preprocessed/index.csv has invalid columns")

    df["image_path"] = df["image_path"].astype(str).map(_normalize_path_string)
    df["mask_path"] = df["mask_path"].astype(str).map(_normalize_path_string)
    valid = df["image_path"].map(os.path.exists) & df["mask_path"].map(os.path.exists)
    df = df[valid].drop_duplicates(subset=["group_id"], keep="last").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No PanNuke image/mask pairs available")

    return {
        "images": df["image_path"].astype(str).to_numpy(),
        "masks": df["mask_path"].astype(str).to_numpy(),
        "labels": df["label_int"].astype(np.int64).to_numpy(),
        "groups": df["group_id"].astype(str).to_numpy(),
    }


def _normalize_path_string(path: str) -> str:
    path = str(path).strip().strip('"').strip("'")
    if os.path.exists(path):
        return path
    alt = path.replace("\\", "/")
    if os.path.exists(alt):
        return alt
    return path


def parse_siim(root: Path) -> dict:
    index_csv = root / "preprocessed" / "index.csv"
    if not index_csv.exists():
        raise FileNotFoundError("SIIM preprocessed/index.csv missing")

    df = pd.read_csv(index_csv)
    required = {"image_path", "mask_path", "label_int", "group_id"}
    if not required.issubset(df.columns):
        raise ValueError("SIIM preprocessed/index.csv has invalid columns")

    df["image_path"] = df["image_path"].astype(str).map(_normalize_path_string)
    df["mask_path"] = df["mask_path"].astype(str).map(_normalize_path_string)

    valid = df["image_path"].map(os.path.exists) & df["mask_path"].map(os.path.exists)
    df = df[valid].drop_duplicates(subset=["group_id"], keep="last").reset_index(drop=True)

    if len(df) < 1000:
        raise RuntimeError(f"SIIM preprocessed index too small after filtering: {len(df)}")

    return {
        "images": df["image_path"].astype(str).to_numpy(),
        "masks": df["mask_path"].astype(str).to_numpy(),
        "labels": df["label_int"].astype(np.int64).to_numpy(),
        "groups": df["group_id"].astype(str).to_numpy(),
    }


def load_dataset_bundle(dataset: str, root: Path) -> dict:
    if dataset == "tcga":
        bundle = parse_tcga(root)
    elif dataset == "isic":
        bundle = parse_isic(root)
    elif dataset == "panda":
        bundle = parse_panda(root)
    elif dataset == "pannuke":
        bundle = parse_pannuke(root)
    elif dataset == "siim":
        bundle = parse_siim(root)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    num_classes = DATASET_META[dataset]["num_classes"]
    labels = bundle["labels"]
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(f"{dataset} labels out of range [0,{num_classes - 1}]")

    return bundle


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        image_paths,
        mask_paths,
        labels,
        seg_classes: int,
        binary_positive_min: int = 1,
        crop_to_mask_bbox: bool = False,
        transform=None,
        cache_size: int = 0,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.seg_classes = seg_classes
        self.binary_positive_min = max(1, int(binary_positive_min))
        self.crop_to_mask_bbox = bool(crop_to_mask_bbox)
        self.transform = transform
        self.cache_size = max(0, int(cache_size))
        self.cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self):
        return len(self.image_paths)

    def _load_pair(self, idx: int):
        if self.cache_size > 0 and idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]

        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))

        if self.cache_size > 0:
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)
            self.cache[idx] = (image, mask)

        return image, mask

    def __getitem__(self, idx):
        image, mask = self._load_pair(idx)

        if self.crop_to_mask_bbox:
            ys, xs = np.where(mask > 0)
            if ys.size > 0 and xs.size > 0:
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                # Small margin to preserve local context around annotated tissue.
                h, w = mask.shape[:2]
                dy = max(1, (y1 - y0) // 20)
                dx = max(1, (x1 - x0) // 20)
                y0 = max(0, y0 - dy)
                y1 = min(h, y1 + dy)
                x0 = max(0, x0 - dx)
                x1 = min(w, x1 + dx)
                image = image[y0:y1, x0:x1]
                mask = mask[y0:y1, x0:x1]

        if self.seg_classes == 1:
            mask = (mask >= self.binary_positive_min).astype(np.float32)
        else:
            mask = mask.astype(np.float32)
            if float(mask.max()) > float(self.seg_classes - 1):
                mask = np.rint(mask * ((self.seg_classes - 1) / 255.0))
            mask = np.clip(mask, 0, self.seg_classes - 1).astype(np.int64)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        if self.seg_classes == 1:
            mask = mask.unsqueeze(0).float()
        else:
            mask = mask.long()

        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return image, mask, label
