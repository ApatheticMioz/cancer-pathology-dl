#!/usr/bin/env python3
"""Standalone probe: Canonical GradNorm (Chen et al., ICML 2018) on PANDA x VGG16.

Features of canonical formulation:
1. Two separate optimizers:
   - optimizer_model: Adam(model.parameters(), lr=1e-3)
   - optimizer_weights: Adam([weights], lr=0.025)
2. Strict gradient detachment:
   - Network parameters do NOT receive gradients from L_grad.
   - Only task weights w receive gradients from L_grad.
3. Weights renormalized after each step: sum(w) = 2.0.
4. Evaluates whether decoupled canonical GradNorm stabilizes fine-grained ISUP grading.

This script runs completely independently and does NOT touch or modify main.py.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_META, DATASET_ROOTS, RANDOM_SEED
from src.data import MultiTaskDataset, build_transforms, load_dataset_bundle, make_group_kfold_splits
from src.metrics import dice_coefficient
from src.models import MultiTaskUNet
from src.utils import append_jsonl, now_iso


def wilson_score_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z ** 2) / total
    centre = (p + (z ** 2) / (2.0 * total)) / denom
    spread = (z * ((p * (1.0 - p) / total + (z ** 2) / (4.0 * total ** 2)) ** 0.5)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)

# v2 probe artifacts live under results/round2/gradnorm_probe/ so the legacy
# log (logs/canonical_gradnorm_run18.log) and checkpoint are never clobbered.
RUN_DIR = PROJECT_ROOT / "results" / "round2" / "gradnorm_probe"
RUN_LABEL = "gradnorm_probe"


def _setup_logging() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RUN_DIR / "gradnorm_probe.log", mode="w"),
        ],
    )


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Canonical GradNorm Probe on PANDA")
    parser.add_argument("--alpha", type=float, default=1.5, help="GradNorm asymmetry exponent (default: 1.5)")
    parser.add_argument("--weight_lr", type=float, default=0.025, help="Learning rate for task weights (default: 0.025)")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs (default: 50)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Model learning rate (default: 1e-3)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    _setup_logging()

    device = torch.device(args.device)
    logger.info("=== Canonical GradNorm (Chen et al. 2018) Probe on PANDA ===")
    logger.info("Device: %s (%s)", device, torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU")
    logger.info("Alpha: %.2f | Weight LR: %.4f | Model LR: %.4f | Batch Size: %d", args.alpha, args.weight_lr, args.lr, args.batch_size)

    # 1. Load PANDA raw dataset (no Macenko)
    root = DATASET_ROOTS["panda"]
    bundle = load_dataset_bundle("panda", root, skip_macenko=True)
    images = bundle["images"]
    masks = bundle["masks"]
    labels = bundle["labels"]
    groups = bundle["groups"]
    meta = DATASET_META["panda"]
    num_classes = meta["num_classes"]
    seg_classes = meta["seg_classes"]

    # 2. Patient-level split via the instrumented splitter (Fold 0, seeded).
    # Routing through make_group_kfold_splits stamps splitter_branch exactly
    # like every campaign run (seed-deterministic, zero-leakage validated).
    splits = make_group_kfold_splits(
        labels, groups, n_splits=5, seed=RANDOM_SEED,
        grouping=meta.get("grouping"),
        provenance_path=root / "preprocessed" / "grouping_provenance.json",
    )
    splitter_branch = splits.metadata.get("branch")
    train_idx, val_idx = splits[0]
    logger.info(
        "Train samples: %d | Val samples: %d | Patient groups: %d | splitter_branch: %s (seed=%d)",
        len(train_idx), len(val_idx), len(np.unique(groups)), splitter_branch, RANDOM_SEED,
    )

    train_tf, val_tf = build_transforms(meta["img_size"])
    train_ds = MultiTaskDataset(images[train_idx], masks[train_idx], labels[train_idx], seg_classes=seg_classes, transform=train_tf)
    val_ds = MultiTaskDataset(images[val_idx], masks[val_idx], labels[val_idx], seg_classes=seg_classes, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Model & Canonical GradNorm Setup
    model = MultiTaskUNet(
        encoder_name="vgg16",
        num_classes=num_classes,
        seg_classes=seg_classes,
        skip_connections=True,
    ).to(device)

    seg_criterion = nn.BCEWithLogitsLoss() if seg_classes == 1 else nn.CrossEntropyLoss()
    cls_criterion = nn.CrossEntropyLoss()

    # Canonical GradNorm: Model optimizer updates model parameters
    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Canonical GradNorm: Learnable task weights parameter w = [w_seg, w_cls]
    # Initialized at static 5:1 normalized to sum=2: w_seg=5/3, w_cls=1/3
    weights = nn.Parameter(torch.tensor([5.0 / 3.0, 1.0 / 3.0], dtype=torch.float32, device=device))
    optimizer_weights = torch.optim.Adam([weights], lr=args.weight_lr)

    # Shared parameters for gradient norm monitoring: encoder layer
    shared_params = [p for p in model.unet.encoder.parameters() if p.requires_grad]

    initial_losses = None
    best_val_score = -1.0
    best_acc = 0.0
    best_dice = 0.0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_seg_loss = 0.0
        train_cls_loss = 0.0
        correct = 0
        total = 0

        for images, masks, targets in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            seg_out, cls_out = model(images)
            l_seg = seg_criterion(seg_out, masks)
            l_cls = cls_criterion(cls_out, targets)

            if initial_losses is None:
                initial_losses = torch.stack([l_seg.detach(), l_cls.detach()])

            # Step 1: Compute task gradient norms wrt shared encoder before optimizer step
            grads_seg = torch.autograd.grad(l_seg, shared_params, retain_graph=True, allow_unused=True)
            grads_cls = torch.autograd.grad(l_cls, shared_params, retain_graph=True, allow_unused=True)

            def _l2_norm(grads):
                valid = [g.norm() for g in grads if g is not None]
                return torch.norm(torch.stack(valid)) if valid else torch.tensor(0.0, device=device)

            norm_seg = _l2_norm(grads_seg)
            norm_cls = _l2_norm(grads_cls)
            norms = torch.stack([norm_seg, norm_cls]).detach()

            # Step 2: Backward pass for model parameters
            w = weights.detach()
            loss_total = w[0] * l_seg + w[1] * l_cls
            optimizer_model.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_model.step()

            # Step 3: Canonical GradNorm update on task weights (optimizer_weights only)
            with torch.no_grad():
                current_losses = torch.stack([l_seg.detach(), l_cls.detach()])
                inv_rates = current_losses / (initial_losses + 1e-8)
                inv_rates = inv_rates / inv_rates.mean().clamp(min=1e-8)
                target_norms = (norms.mean() * (inv_rates ** args.alpha)).detach()

            grad_loss = torch.sum(torch.abs(weights * norms - target_norms))
            optimizer_weights.zero_grad(set_to_none=True)
            grad_loss.backward()
            optimizer_weights.step()

            # Renormalize weights so sum(w) = 2.0
            with torch.no_grad():
                weights.data = weights.data.clamp(min=1e-4)
                weights.data = weights.data * (2.0 / weights.data.sum())

            train_loss += loss_total.item()
            train_seg_loss += l_seg.item()
            train_cls_loss += l_cls.item()
            preds = cls_out.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_acc = 100.0 * correct / max(total, 1)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_dices = []

        with torch.no_grad():
            for images, masks, targets in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                targets = targets.to(device)

                seg_out, cls_out = model(images)
                preds = cls_out.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                val_dices.append(dice_coefficient(seg_out, masks, seg_classes))

        val_acc = 100.0 * val_correct / max(val_total, 1)
        val_dice = 100.0 * np.mean(val_dices)
        score = 0.5 * (val_acc + val_dice)

        low_ci, high_ci = wilson_score_interval(val_correct, val_total)
        logger.info(
            "Epoch %02d/%02d | Loss: %.4f | Train Acc: %.2f%% | Val Acc: %.2f%% [%.2f, %.2f] | Val Dice: %.2f%% | Weights: [seg=%.3f, cls=%.3f]",
            epoch, args.epochs, train_loss / len(train_loader), train_acc, val_acc, 100.0 * low_ci, 100.0 * high_ci, val_dice, weights[0].item(), weights[1].item()
        )

        # Per-epoch telemetry in the per-run epoch_log.jsonl contract
        # (run_label/fold/seed/splitter_branch stamped like campaign runs).
        append_jsonl(RUN_DIR / "epoch_log.jsonl", {
            "timestamp": now_iso(),
            "dataset": "panda",
            "encoder": "vgg16",
            "epoch": epoch,
            "batch_size": args.batch_size,
            "tr_loss": round(train_loss / max(len(train_loader), 1), 6),
            "tr_acc": round(train_acc, 6),
            "vl_acc": round(val_acc, 6),
            "vl_dice": round(val_dice, 6),
            "w_seg": round(float(weights[0].item()), 6),
            "w_cls": round(float(weights[1].item()), 6),
            "run_label": RUN_LABEL,
            "fold": 0,
            "seed": int(RANDOM_SEED),
            "splitter_branch": splitter_branch,
        })

        if score > best_val_score:
            best_val_score = score
            best_acc = val_acc
            best_dice = val_dice
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "val_dice": val_dice,
                    "weights": weights.cpu().tolist(),
                },
                RUN_DIR / "best.pth",
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered after %d epochs.", epoch)
                break

    elapsed = time.time() - start_time
    logger.info("=======================================================")
    logger.info("Canonical GradNorm Probe Completed in %.1f minutes", elapsed / 60.0)
    logger.info("Best Val Acc: %.2f%% | Best Val Dice: %.2f%%", best_acc, best_dice)
    logger.info("Baseline Static (Run 20): 45.39%% Acc | 44.08%% Dice")
    logger.info("Coupled Adam GradNorm (Run 18): 29.04%% Acc | 31.43%% Dice")
    logger.info("=======================================================")


if __name__ == "__main__":
    main()
