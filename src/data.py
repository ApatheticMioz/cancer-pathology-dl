"""Dataset loading, transforms, and group-aware splitting.

Provides:
    - MultiTaskDataset: PyTorch dataset with optional RAM caching.
    - build_transforms: Train/val albumentation pipelines.
    - make_group_split: GroupShuffleSplit or StratifiedShuffleSplit
      depending on group cardinality.
    - Dataset parsers for TCGA, PANDA, SIIM, and PanNuke.
"""
from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from torch.utils.data import Dataset

from src.config import DATASET_META
from src.utils import first_existing

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(img_size: int):
    """Build train and validation albumentation pipelines.

    Args:
        img_size: Target square dimension for resizing.

    Returns:
        Tuple of (train_transform, val_transform).
    """
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

# ---------------------------------------------------------------------------
# Train/Val splitting (Group-aware & 5-Fold CV with Zero-Leakage Guarantee)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Split result containers (carry branch metadata for persistence)
# ---------------------------------------------------------------------------

class GroupKFoldSplits(list):
    """K-fold split list that also records the executed branch metadata.

    Behaves as a plain ``list`` of ``(train_idx, val_idx)`` tuples so existing
    callers (``splits[fold_idx]``) keep working, but additionally exposes
    ``.metadata`` so training can persist which splitter branch was executed.
    """

    def __new__(cls, splits, metadata):
        obj = super().__new__(cls)
        obj.extend(splits)
        obj.metadata = metadata
        return obj

    def __init__(self, splits, metadata):
        # __new__ already populated the list and metadata; no-op here so the
        # inherited list.__init__ (which accepts at most 1 arg) is not called.
        pass


class GroupSplitResult(tuple):
    """A ``(train_idx, val_idx)`` pair that also records branch metadata.

    Behaves as a plain 2-tuple so existing callers (``tr, vl = ...``) keep
    working, but additionally exposes ``.metadata`` for persistence.
    """

    def __new__(cls, train_idx, val_idx, metadata):
        obj = super().__new__(cls, (train_idx, val_idx))
        obj.metadata = metadata
        return obj

    def __init__(self, train_idx, val_idx, metadata):
        # __new__ already built the 2-tuple and metadata; no-op here so the
        # inherited tuple.__init__ (which accepts at most 1 arg) is not called.
        pass


def _load_grouping_provenance(provenance_path):
    """Load the grouping-provenance sidecar if it exists.

    The sidecar (``grouping_provenance.json``) is written next to the dataset
    index by the preprocessing builder and records the *semantic source* of the
    ``group_id`` column (e.g. ``"PatientID"``) plus cardinality facts. It lets
    the splitters distinguish a *legitimate* degenerate cardinality (a release
    that genuinely has one image per patient) from an *unproven* one.

    Returns the parsed dict, or ``None`` if the path is unset / missing / unreadable.
    """
    if provenance_path is None:
        return None
    p = Path(provenance_path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read grouping provenance sidecar %s: %s", p, exc)
        return None


def _validate_grouping(declared, n_samples, n_unique_groups, provenance=None):
    """Validate a declared grouping unit against actual group cardinality.

    Returns a *verdict* string that the caller uses to pick the executed branch:
      - ``None``: normal path (non-trivial groups, or ``image``/``patch`` trivial
        groups). The caller uses its standard branch.
      - ``"trivial_patient"``: declared ``'patient'`` with degenerate cardinality
        (``n_unique_groups == n_samples``) that is PROVEN legitimate by a
        ``grouping_provenance.json`` sidecar whose ``source == "PatientID"``.
        This is a data property (the release genuinely has one image per
        patient), so the caller must use the loud, recorded trivial-patient
        branch — NOT fatal.

    Raises:
      - ``ValueError``: unknown declaration.
      - ``RuntimeError`` (FATAL): declared ``'patient'`` with degenerate
        cardinality and NO provenance sidecar proving the groups are
        patient-derived (an unproven claim).
    """
    if declared is None:
        return None
    if declared not in {"patient", "image", "patch"}:
        raise ValueError(f"Unknown grouping declaration: {declared!r}")
    if declared == "patient" and n_unique_groups == n_samples:
        if provenance is not None and provenance.get("source") == "PatientID":
            return "trivial_patient"
        raise RuntimeError(
            "FATAL GROUPING DEFECT: declared grouping='patient' but "
            f"n_unique_groups ({n_unique_groups}) == n_samples ({n_samples}); "
            "every sample is its own group, so group-aware splitting is vacuous, "
            "and no grouping_provenance.json sidecar proves the groups are "
            "patient-derived. Regenerate the index at patient level (emitting the "
            "provenance sidecar) or correct the declaration."
        )
    return None


def _check_fold_class_coverage(splits, labels):
    """Assert every fold's val set contains >=1 sample of every overall class.

    Raises FATAL with a per-fold class-count table on violation. Returns the
    per-fold class-count table (dict) for metadata persistence.
    """
    all_classes = sorted(int(c) for c in np.unique(labels))
    table = {}
    for fold_idx, (tr_idx, vl_idx) in enumerate(splits):
        val_counts = {c: 0 for c in all_classes}
        for lab in labels[vl_idx]:
            val_counts[int(lab)] += 1
        table[f"fold{fold_idx + 1}"] = val_counts
        missing = [c for c in all_classes if val_counts[c] == 0]
        if missing:
            lines = [
                "FATAL CLASS-COVERAGE DEFECT: a fold's val set is missing one or more overall classes."
            ]
            lines.append(f"  overall classes: {all_classes}")
            for f, counts in table.items():
                lines.append(f"  {f}: " + ", ".join(f"class {c}={counts[c]}" for c in all_classes))
            lines.append(f"  missing in fold {fold_idx + 1}: {missing}")
            raise RuntimeError("\n".join(lines))
    return table


def make_group_kfold_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    grouping: str | None = None,
    provenance_path: str | Path | None = None,
) -> GroupKFoldSplits:
    """Create group-aware K-Fold cross-validation splits with validated grouping.

    Guarantees strict zero-leakage across folds (no group spans train and val of
    any fold) and validates the declared grouping unit against the actual group
    cardinality. Every executed branch is explicitly logged and recorded in the
    returned object's ``.metadata`` so training can persist it.

    Args:
        labels: Classification labels array.
        groups: Group identifiers (e.g. patient ID or slide ID).
        n_splits: Number of folds (default 5).
        seed: Random seed for reproducible shuffling.
        grouping: Declared grouping unit from ``DATASET_META`` ('patient',
            'image', or 'patch'). ``None`` skips declaration validation but the
            executed branch is still logged and recorded.
        provenance_path: Optional path to a ``grouping_provenance.json`` sidecar
            next to the dataset index. When present and its ``source`` is
            ``"PatientID"``, a degenerate (one-group-per-sample) cardinality for
            a declared-'patient' dataset is treated as a legitimate data property
            (loud, recorded trivial-patient branch) instead of a FATAL defect.

    Returns:
        ``GroupKFoldSplits``: a list of (train_indices, val_indices) tuples of
        length ``n_splits`` with a ``.metadata`` dict recording the executed
        branch, declaration, provenance, and per-fold class coverage.
    """
    n_samples = len(labels)
    unique_groups = np.unique(groups)
    n_unique_groups = len(unique_groups)

    provenance = _load_grouping_provenance(provenance_path)
    verdict = _validate_grouping(grouping, n_samples, n_unique_groups, provenance)

    metadata = {
        "n_samples": n_samples,
        "n_unique_groups": n_unique_groups,
        "declared_grouping": grouping,
        "n_splits": n_splits,
        "seed": seed,
        "grouping_provenance": provenance,
    }

    if n_unique_groups == n_samples:
        if verdict == "trivial_patient":
            # Legitimate degenerate cardinality: the release genuinely has one
            # image per patient (proven by the sidecar). Loud + recorded, NOT
            # fatal. Group-aware splitting is vacuous here, so we stratify on
            # class only — but we say so explicitly.
            branch = "stratified_kfold_trivial_patient_groups"
            logger.warning(
                "make_group_kfold_splits: declared grouping='patient' but the release "
                "has exactly 1 image per patient (n_unique_groups == n_samples == %d). "
                "Provenance sidecar source=%r confirms this is a data property, not a "
                "misdeclaration. Using %s (group-aware splitting is vacuous; class "
                "stratification only).",
                n_samples, provenance.get("source") if provenance else None, branch,
            )
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(splitter.split(np.zeros(n_samples), labels))
        else:
            # Trivial groups: every sample is its own group. Only reachable when
            # grouping is 'image'/'patch' (or None); a 'patient' declaration
            # without a provenance sidecar would have FATALed above.
            branch = "StratifiedKFold (trivial-group fallback)"
            logger.warning(
                "make_group_kfold_splits: trivial groups (n_unique_groups == n_samples == %d); "
                "using %s. declared_grouping=%r",
                n_samples, branch, grouping,
            )
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(splitter.split(np.zeros(n_samples), labels))
    else:
        # Attempt StratifiedGroupKFold for balanced class-group partitioning
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(splitter.split(np.zeros(n_samples), labels, groups))
            branch = "StratifiedGroupKFold"
        except Exception as exc:
            logger.warning(
                "make_group_kfold_splits: StratifiedGroupKFold failed (%s: %s); "
                "falling back to GroupKFold.",
                type(exc).__name__, exc,
            )
            splitter = GroupKFold(n_splits=n_splits)
            splits = list(splitter.split(np.zeros(n_samples), labels, groups))
            branch = "GroupKFold (fallback from StratifiedGroupKFold)"

    metadata["branch"] = branch

    # Strict zero-leakage mathematical assertion (stays FATAL)
    for fold_idx, (tr_idx, vl_idx) in enumerate(splits):
        tr_g = set(groups[tr_idx])
        vl_g = set(groups[vl_idx])
        overlap = tr_g & vl_g
        if overlap:
            raise RuntimeError(
                f"FATAL LEAKAGE DEFECT: Fold {fold_idx} has {len(overlap)} overlapping groups! "
                f"Samples in train={len(tr_idx)}, val={len(vl_idx)}."
            )
        logger.debug(
            "Fold %d/%d: train=%d samples (%d groups), val=%d samples (%d groups), overlap=0",
            fold_idx + 1, n_splits, len(tr_idx), len(tr_g), len(vl_idx), len(vl_g)
        )

    # Per-fold class coverage (FATAL with counts table on violation)
    metadata["per_fold_class_coverage"] = _check_fold_class_coverage(splits, labels)

    return GroupKFoldSplits(splits, metadata)


def make_group_split(
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int = 42,
    test_size: float = 0.2,
    grouping: str | None = None,
    provenance_path: str | Path | None = None,
) -> GroupSplitResult:
    """Create a single group-aware or stratified train/val split with validated grouping.

    Uses GroupShuffleSplit when groups have meaningful cardinality (fewer unique
    groups than samples). Falls back to StratifiedShuffleSplit when every sample
    has a unique group identifier. The declared grouping unit is validated
    against the actual group cardinality, and the executed branch is explicitly
    logged and recorded in the returned object's ``.metadata`` for persistence.

    Args:
        labels: Classification labels.
        groups: Group identifiers (e.g., patient ID).
        seed: Random seed for reproducibility.
        test_size: Fraction reserved for validation.
        grouping: Declared grouping unit from ``DATASET_META`` ('patient',
            'image', or 'patch'). ``None`` skips declaration validation but the
            executed branch is still logged and recorded.
        provenance_path: Optional path to a ``grouping_provenance.json`` sidecar
            next to the dataset index. When present and its ``source`` is
            ``"PatientID"``, a degenerate (one-group-per-sample) cardinality for
            a declared-'patient' dataset is treated as a legitimate data property
            (loud, recorded trivial-patient branch) instead of a FATAL defect.

    Returns:
        ``GroupSplitResult``: a ``(train_indices, val_indices)`` pair with a
        ``.metadata`` dict recording the executed branch, declaration, and
        provenance.
    """
    n_samples = len(labels)
    n_unique_groups = len(np.unique(groups))

    provenance = _load_grouping_provenance(provenance_path)
    verdict = _validate_grouping(grouping, n_samples, n_unique_groups, provenance)

    metadata = {
        "n_samples": n_samples,
        "n_unique_groups": n_unique_groups,
        "declared_grouping": grouping,
        "seed": seed,
        "test_size": test_size,
        "grouping_provenance": provenance,
    }

    if n_unique_groups == n_samples:
        if verdict == "trivial_patient":
            # Legitimate degenerate cardinality: the release genuinely has one
            # image per patient (proven by the sidecar). Loud + recorded, NOT
            # fatal. Group-aware splitting is vacuous here, so we stratify on
            # class only — but we say so explicitly.
            branch = "stratified_shufflesplit_trivial_patient_groups"
            logger.warning(
                "make_group_split: declared grouping='patient' but the release has "
                "exactly 1 image per patient (n_unique_groups == n_samples == %d). "
                "Provenance sidecar source=%r confirms this is a data property, not a "
                "misdeclaration. Using %s (group-aware splitting is vacuous; class "
                "stratification only).",
                n_samples, provenance.get("source") if provenance else None, branch,
            )
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            splits = list(splitter.split(np.zeros(n_samples), labels))
            if not splits:
                raise RuntimeError("Could not create stratified split")
            tr_idx, vl_idx = splits[0]
        else:
            # Trivial groups: every sample is its own group. Only reachable when
            # grouping is 'image'/'patch' (or None); a 'patient' declaration
            # without a provenance sidecar would have FATALed above.
            branch = "StratifiedShuffleSplit (trivial-group fallback)"
            logger.warning(
                "make_group_split: trivial groups (n_unique_groups == n_samples == %d); "
                "using %s. declared_grouping=%r",
                n_samples, branch, grouping,
            )
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            splits = list(splitter.split(np.zeros(n_samples), labels))
            if not splits:
                raise RuntimeError("Could not create stratified split")
            tr_idx, vl_idx = splits[0]
    else:
        branch = "GroupShuffleSplit"
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        splits = list(splitter.split(np.zeros(n_samples), labels, groups))
        if not splits:
            raise RuntimeError("Could not create group split")
        tr_idx, vl_idx = splits[0]

    metadata["branch"] = branch

    overlap = set(groups[tr_idx]) & set(groups[vl_idx])
    if overlap:
        raise RuntimeError(f"FATAL LEAKAGE DEFECT: make_group_split has {len(overlap)} overlapping groups!")

    return GroupSplitResult(tr_idx, vl_idx, metadata)

# ---------------------------------------------------------------------------
# Dataset parsers
# ---------------------------------------------------------------------------

def _normalize_path_string(path: str) -> str:
    """Normalize path separators and strip quotes."""
    path = str(path).strip().strip('"').strip("'")
    if os.path.exists(path):
        return path
    alt = path.replace("\\", "/")
    if os.path.exists(alt):
        return alt
    return path


def _index_stems(directory: Path, patterns: list[str]) -> dict[str, str]:
    """Map file stems to full paths for files matching *patterns*."""
    mapping = {}
    for pattern in patterns:
        for p in directory.rglob(pattern):
            if p.is_file():
                mapping[p.stem] = str(p)
    return mapping


def parse_tcga(root: Path) -> dict:
    """Parse TCGA WSI tiles from patient subdirectories.

    Expects ``root/<patient>/<image>.tif`` with matching ``<image>_mask.tif``.
    """
    images, masks, labels, groups = [], [], [], []
    for patient_dir in sorted(d for d in root.iterdir() if d.is_dir()):
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

    logger.info("TCGA: %d pairs loaded", len(images))
    return {
        "images": np.array(images),
        "masks": np.array(masks),
        "labels": np.array(labels, dtype=np.int64),
        "groups": np.array(groups),
    }


def parse_panda(root: Path, skip_macenko: bool = False) -> dict:
    """Parse PANDA prostate images using train.csv.

    Args:
        root: Dataset root directory.
        skip_macenko: If True, load raw images instead of Macenko-normalized
            images (ablation study for domain shift impact).
    """
    train_csv = root / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError("PANDA train.csv missing")

    # Select image directory based on Macenko ablation flag
    if skip_macenko:
        image_dir = first_existing([
            root / "train_images",
            root / "images",
        ])
        logger.info("PANDA: loading RAW images (Macenko normalization bypassed)")
    else:
        image_dir = root / "preprocessed_macenko" / "images"

    mask_dir = first_existing([root / "train_label_masks" / "train_label_masks", root / "train_label_masks"])
    if image_dir is None or mask_dir is None:
        raise FileNotFoundError("PANDA train_images/train_label_masks directory missing")
    if not image_dir.exists():
        raise FileNotFoundError(f"PANDA image directory missing: {image_dir}")

    image_map = _index_stems(image_dir, ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"])
    mask_map = _index_stems(mask_dir, ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"])
    mask_lookup = {
        (stem[:-5] if stem.endswith("_mask") else stem): path
        for stem, path in mask_map.items()
    }

    df = pd.read_csv(train_csv)
    if "image_id" not in df.columns or "isup_grade" not in df.columns:
        raise ValueError("PANDA train.csv missing image_id/isup_grade")

    images, masks, labels, groups = [], [], [], []
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
        if not (0 <= isup_grade <= 5):
            continue
        images.append(image_path)
        masks.append(mask_path)
        labels.append(isup_grade)
        groups.append(image_id)

    if not images:
        raise RuntimeError("No PANDA image/mask pairs matched train.csv")

    logger.info("PANDA: %d pairs loaded", len(images))
    return {
        "images": np.array(images),
        "masks": np.array(masks),
        "labels": np.array(labels, dtype=np.int64),
        "groups": np.array(groups),
    }


def parse_siim(root: Path) -> dict:
    """Parse SIIM pneumothorax from preprocessed index.csv.

    Uses explicit, hardcoded directory structure relative to *root*:
        images -> root/preprocessed/images/<filename>
        masks  -> root/preprocessed/masks/<filename>
    """
    index_csv = root / "preprocessed" / "index.csv"
    if not index_csv.exists():
        raise FileNotFoundError("SIIM preprocessed/index.csv missing")

    df = pd.read_csv(index_csv)
    required = {"image_path", "mask_path", "label_int", "group_id"}
    if not required.issubset(df.columns):
        raise ValueError("SIIM preprocessed/index.csv has invalid columns")

    # Build absolute paths from filenames only
    img_dir = root / "preprocessed" / "images"
    mask_dir = root / "preprocessed" / "masks"

    df["image_path"] = df["image_path"].astype(str).map(
        lambda p: str(img_dir / Path(p).name)
    )
    df["mask_path"] = df["mask_path"].astype(str).map(
        lambda p: str(mask_dir / Path(p).name)
    )

    valid = df["image_path"].map(os.path.exists) & df["mask_path"].map(os.path.exists)
    df = df[valid].drop_duplicates(subset=["group_id"], keep="last").reset_index(drop=True)

    if len(df) == 0:
        # Debug: show exactly what we were looking for
        first_img = df.iloc[0]["image_path"] if len(df) > 0 else "N/A"
        first_mask = df.iloc[0]["mask_path"] if len(df) > 0 else "N/A"
        # Reconstruct from raw CSV for debug
        raw_df = pd.read_csv(index_csv)
        raw_df["_img"] = raw_df["image_path"].astype(str).map(lambda p: str(img_dir / Path(p).name))
        raw_df["_mask"] = raw_df["mask_path"].astype(str).map(lambda p: str(mask_dir / Path(p).name))
        print(f"DEBUG SIIM: Could not find any valid pairs. First attempted paths:")
        print(f"  IMAGE: {raw_df['_img'].iloc[0]}")
        print(f"  MASK:  {raw_df['_mask'].iloc[0]}")
        print(f"  img_dir exists: {img_dir.exists()}, mask_dir exists: {mask_dir.exists()}")
        raise RuntimeError(f"SIIM preprocessed index too small after filtering: {len(df)}")

    if len(df) < 1000:
        raise RuntimeError(f"SIIM preprocessed index too small after filtering: {len(df)}")

    logger.info("SIIM: %d pairs loaded", len(df))
    return {
        "images": df["image_path"].astype(str).to_numpy(),
        "masks": df["mask_path"].astype(str).to_numpy(),
        "labels": df["label_int"].astype(np.int64).to_numpy(),
        "groups": df["group_id"].astype(str).to_numpy(),
    }


def parse_pannuke(root: Path, skip_macenko: bool = False) -> dict:
    """Parse PanNuke from preprocessed index.csv.

    Uses explicit, hardcoded directory structure relative to *root*:
        images (raw)      -> root/preprocessed/images/<filename>
        images (macenko)  -> root/preprocessed_macenko/images/<filename>
        masks             -> root/preprocessed/masks/<filename>

    Args:
        root: Dataset root directory.
        skip_macenko: If True, load raw images from the original paths in
            index.csv instead of Macenko-normalized images (ablation study
            for domain shift impact).
    """
    index_csv = root / "preprocessed" / "index.csv"
    if not index_csv.exists():
        raise FileNotFoundError("PanNuke preprocessed/index.csv missing")

    df = pd.read_csv(index_csv)
    required = {"image_path", "mask_path", "label_int", "group_id"}
    if not required.issubset(df.columns):
        raise ValueError("PanNuke preprocessed/index.csv has invalid columns")

    mask_dir = root / "preprocessed" / "masks"

    if skip_macenko:
        logger.info("PanNuke: loading RAW images (Macenko normalization bypassed)")
        img_dir = root / "preprocessed" / "images"
    else:
        img_dir = root / "preprocessed_macenko" / "images"
        if not img_dir.exists():
            raise FileNotFoundError("PanNuke preprocessed_macenko/images missing")

    # Build absolute paths from filenames only
    df["image_path"] = df["image_path"].astype(str).map(
        lambda p: str(img_dir / Path(p).name)
    )
    df["mask_path"] = df["mask_path"].astype(str).map(
        lambda p: str(mask_dir / Path(p).name)
    )

    valid = df["image_path"].map(os.path.exists) & df["mask_path"].map(os.path.exists)
    df = df[valid].drop_duplicates(subset=["group_id"], keep="last").reset_index(drop=True)

    if len(df) == 0:
        # Debug: show exactly what we were looking for
        raw_df = pd.read_csv(index_csv)
        raw_df["_img"] = raw_df["image_path"].astype(str).map(lambda p: str(img_dir / Path(p).name))
        raw_df["_mask"] = raw_df["mask_path"].astype(str).map(lambda p: str(mask_dir / Path(p).name))
        print(f"DEBUG PanNuke: Could not find any valid pairs. First attempted paths:")
        print(f"  IMAGE: {raw_df['_img'].iloc[0]}")
        print(f"  MASK:  {raw_df['_mask'].iloc[0]}")
        print(f"  img_dir exists: {img_dir.exists()}, mask_dir exists: {mask_dir.exists()}")
        raise RuntimeError(f"PanNuke preprocessed index too small after filtering: {len(df)}")

    if len(df) < 4000:
        raise RuntimeError(f"PanNuke preprocessed index too small after filtering: {len(df)}")

    logger.info("PanNuke: %d pairs loaded", len(df))
    return {
        "images": df["image_path"].astype(str).to_numpy(),
        "masks": df["mask_path"].astype(str).to_numpy(),
        "labels": df["label_int"].astype(np.int64).to_numpy(),
        "groups": df["group_id"].astype(str).to_numpy(),
    }


def load_dataset_bundle(dataset: str, root: Path, skip_macenko: bool = False) -> dict:
    """Load and validate a full dataset bundle.

    Args:
        dataset: One of 'tcga', 'panda', 'siim', 'pannuke'.
        root: Path to the dataset root directory.
        skip_macenko: If True, bypass Macenko normalization for datasets
            that support it (panda, pannuke). Tests domain shift impact.

    Returns:
        Dict with keys: images, masks, labels, groups.
    """
    if skip_macenko and dataset in {"panda", "pannuke"}:
        logger.info(
            "%s: Macenko normalization bypassed (--no-macenko flag active)",
            dataset.upper(),
        )

    # Router: parsers that accept the skip_macenko flag
    if dataset == "panda":
        bundle = parse_panda(root, skip_macenko=skip_macenko)
    elif dataset == "pannuke":
        bundle = parse_pannuke(root, skip_macenko=skip_macenko)
    elif dataset == "tcga":
        bundle = parse_tcga(root)
    elif dataset == "siim":
        bundle = parse_siim(root)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    num_classes = DATASET_META[dataset]["num_classes"]
    labels = bundle["labels"]
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(f"{dataset} labels out of range [0, {num_classes - 1}]")
    return bundle

# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class MultiTaskDataset(Dataset):
    """Multi-task dataset yielding (image, segmentation_mask, classification_label).

    Supports optional LRU caching of decoded image/mask pairs to reduce
    disk I/O during training.

    Args:
        image_paths: Array of image file paths.
        mask_paths: Array of mask file paths.
        labels: Array of integer classification labels.
        seg_classes: Number of segmentation classes (1 = binary).
        binary_positive_min: Threshold for binary mask binarization.
        crop_to_mask_bbox: If True, crop to mask bounding box before transforms.
        transform: Albumentations compose transform.
        cache_size: Number of decoded pairs to keep in RAM (0 = disabled).
    """

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
        case_ids=None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        # Case identifiers for per-sample result dumps. Defaults to the image
        # filename stem (stable, unique per sample); callers may override.
        if case_ids is None:
            self.case_ids = [
                Path(str(p)).stem for p in image_paths
            ]
        else:
            self.case_ids = list(case_ids)
        self.seg_classes = seg_classes
        self.binary_positive_min = max(1, int(binary_positive_min))
        self.crop_to_mask_bbox = bool(crop_to_mask_bbox)
        self.transform = transform
        self.cache_size = max(0, int(cache_size))
        self.cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.image_paths)

    def case_id(self, idx: int) -> str:
        """Return the case identifier for sample ``idx`` (for result dumps)."""
        return str(self.case_ids[idx])

    def _load_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Load (image, mask) pair, using cache if enabled."""
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, mask = self._load_pair(idx)

        # Optional bounding-box crop around annotated region
        if self.crop_to_mask_bbox:
            ys, xs = np.where(mask > 0)
            if ys.size > 0 and xs.size > 0:
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                h, w = mask.shape[:2]
                dy = max(1, (y1 - y0) // 20)
                dx = max(1, (x1 - x0) // 20)
                y0 = max(0, y0 - dy)
                y1 = min(h, y1 + dy)
                x0 = max(0, x0 - dx)
                x1 = min(w, x1 + dx)
                image = image[y0:y1, x0:x1]
                mask = mask[y0:y1, x0:x1]

        # Mask encoding
        if self.seg_classes == 1:
            mask = (mask >= self.binary_positive_min).astype(np.float32)
        else:
            mask = mask.astype(np.float32)
            if float(mask.max()) > float(self.seg_classes - 1):
                mask = np.rint(mask * ((self.seg_classes - 1) / 255.0))
            mask = np.clip(mask, 0, self.seg_classes - 1).astype(np.int64)

        # Augmentation
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        # Tensor shaping
        if self.seg_classes == 1:
            mask = mask.unsqueeze(0).float()
        else:
            mask = mask.long()

        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return image, mask, label