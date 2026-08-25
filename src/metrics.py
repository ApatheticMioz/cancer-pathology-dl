"""Metric calculations and diagnostic evaluation for multi-task cancer imaging.

Provides:
    - dice_coefficient: Mean Dice coefficient over a batch (with configurable empty score).
    - positive_slice_dice: Dice computed strictly on samples with non-empty ground-truth masks.
    - iou_coefficient: Jaccard Index (Intersection over Union).
    - compute_comprehensive_metrics: Full suite of segmentation and classification metrics.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def dice_coefficient(
    seg_pred: torch.Tensor,
    seg_target: torch.Tensor,
    seg_classes: int,
    empty_score: float = 1.0,
) -> float:
    """Compute mean Dice coefficient over a batch.

    Args:
        seg_pred: Raw segmentation logits (B, C, H, W) or (B, 1, H, W).
        seg_target: Ground-truth masks.
        seg_classes: Number of segmentation classes (1 = binary).
        empty_score: Value assigned when both pred and target are empty (union == 0).
                     Default is 1.0 (historical convention), set to 0.0 or np.nan for diagnostic analysis.

    Returns:
        Mean Dice score as a Python float.
    """
    if seg_classes == 1:
        pred = (torch.sigmoid(seg_pred) > 0.5).float()
        intersection = (pred * seg_target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + seg_target.sum(dim=(1, 2, 3))
        empty_val = torch.full_like(union, fill_value=empty_score)
        scores = torch.where(
            union == 0,
            empty_val,
            (2.0 * intersection) / (union + 1e-8),
        )
        return float(scores.mean().item())

    pred = torch.argmax(seg_pred, dim=1)
    scores = []
    for c in range(seg_classes):
        p = (pred == c).float()
        t = (seg_target == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        if float(union) == 0.0:
            scores.append(float(empty_score))
        else:
            scores.append(float((2.0 * inter) / (union + 1e-8)))
    return float(np.mean(scores))


def positive_slice_dice(
    seg_pred: torch.Tensor,
    seg_target: torch.Tensor,
    seg_classes: int,
) -> float | None:
    """Compute mean Dice coefficient strictly on samples/slices that contain foreground lesions.

    Filters out empty ground-truth masks (sum == 0) to eliminate true-negative empty mask inflation.

    Args:
        seg_pred: Raw segmentation logits.
        seg_target: Ground-truth masks.
        seg_classes: Number of segmentation classes.

    Returns:
        Mean foreground Dice score, or None if no positive slices exist in the batch.
    """
    if seg_classes == 1:
        target_sums = seg_target.sum(dim=(1, 2, 3))
        pos_mask = target_sums > 0
        if not pos_mask.any():
            return None
        pred = (torch.sigmoid(seg_pred[pos_mask]) > 0.5).float()
        target = seg_target[pos_mask]
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        scores = (2.0 * intersection) / (union + 1e-8)
        return float(scores.mean().item())

    # Multi-class: compute foreground-only dice (classes >= 1)
    pred = torch.argmax(seg_pred, dim=1)
    scores = []
    for c in range(1, seg_classes):
        p = (pred == c).float()
        t = (seg_target == c).float()
        if float(t.sum()) == 0.0:
            continue
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        scores.append(float((2.0 * inter) / (union + 1e-8)))

    if not scores:
        return None
    return float(np.mean(scores))


def iou_coefficient(
    seg_pred: torch.Tensor,
    seg_target: torch.Tensor,
    seg_classes: int,
) -> float:
    """Compute Intersection over Union (Jaccard Index) over a batch."""
    if seg_classes == 1:
        pred = (torch.sigmoid(seg_pred) > 0.5).float()
        intersection = (pred * seg_target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + seg_target.sum(dim=(1, 2, 3)) - intersection
        scores = torch.where(
            union == 0,
            torch.ones_like(union),
            intersection / (union + 1e-8),
        )
        return float(scores.mean().item())

    pred = torch.argmax(seg_pred, dim=1)
    scores = []
    for c in range(seg_classes):
        p = (pred == c).float()
        t = (seg_target == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        if float(union) == 0.0:
            scores.append(1.0)
        else:
            scores.append(float(inter / (union + 1e-8)))
    return float(np.mean(scores))


def compute_comprehensive_metrics(
    seg_pred: torch.Tensor,
    seg_target: torch.Tensor,
    cls_pred: torch.Tensor,
    cls_target: torch.Tensor,
    seg_classes: int,
    num_classes: int,
) -> dict[str, float]:
    """Compute full suite of multi-task diagnostic metrics.

    Returns dict with:
        - dice_all: Standard macro Dice (empty=1.0)
        - dice_pos: Positive-slice foreground Dice (empty excluded)
        - iou: Jaccard index
        - cls_acc: Top-1 classification accuracy
        - cls_balanced_acc: Mean per-class recall
        - empty_mask_pct: Percentage of samples with zero foreground in ground truth
    """
    metrics: dict[str, float] = {}

    # Segmentation
    metrics["dice_all"] = dice_coefficient(seg_pred, seg_target, seg_classes, empty_score=1.0)
    metrics["dice_zero_empty"] = dice_coefficient(seg_pred, seg_target, seg_classes, empty_score=0.0)
    pos_dice = positive_slice_dice(seg_pred, seg_target, seg_classes)
    metrics["dice_pos"] = pos_dice if pos_dice is not None else float("nan")
    metrics["iou"] = iou_coefficient(seg_pred, seg_target, seg_classes)

    if seg_classes == 1:
        target_sums = seg_target.sum(dim=(1, 2, 3))
        empty_count = int((target_sums == 0).sum().item())
        metrics["empty_mask_pct"] = (empty_count / max(1, len(seg_target))) * 100.0
    else:
        # Multi-class: check if all foreground classes are 0
        fg_sum = (seg_target > 0).float().sum(dim=(1, 2) if seg_target.ndim == 3 else (1, 2, 3))
        empty_count = int((fg_sum == 0).sum().item())
        metrics["empty_mask_pct"] = (empty_count / max(1, len(seg_target))) * 100.0

    # Classification
    pred_labels = cls_pred.argmax(dim=1)
    correct = (pred_labels == cls_target).float()
    metrics["cls_acc"] = float(correct.mean().item()) * 100.0

    # Balanced accuracy (mean recall per class)
    per_class_recalls = []
    for c in range(num_classes):
        c_mask = cls_target == c
        if c_mask.any():
            rec = (pred_labels[c_mask] == c).float().mean().item()
            per_class_recalls.append(rec)
    metrics["cls_balanced_acc"] = (float(np.mean(per_class_recalls)) * 100.0) if per_class_recalls else 0.0

    return metrics