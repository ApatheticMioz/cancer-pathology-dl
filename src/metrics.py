"""Metric calculations for multi-task training.

Provides:
    - dice_coefficient: Mean Dice coefficient over a batch.
"""
from __future__ import annotations

import numpy as np
import torch


def dice_coefficient(seg_pred: torch.Tensor, seg_target: torch.Tensor, seg_classes: int) -> float:
    """Compute mean Dice coefficient over a batch.

    Args:
        seg_pred: Raw segmentation logits (B, C, H, W) or (B, 1, H, W).
        seg_target: Ground-truth masks.
        seg_classes: Number of segmentation classes.

    Returns:
        Mean Dice score as a Python float.
    """
    if seg_classes == 1:
        pred = (torch.sigmoid(seg_pred) > 0.5).float()
        intersection = (pred * seg_target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + seg_target.sum(dim=(1, 2, 3))
        scores = torch.where(
            union == 0,
            torch.ones_like(union),
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
            scores.append(1.0)
        else:
            scores.append(float((2.0 * inter) / (union + 1e-8)))
    return float(np.mean(scores))