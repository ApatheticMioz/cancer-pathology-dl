"""Epoch loops and training orchestrator.

Provides:
    - _run_epoch: Single training or validation epoch.
    - train_single_run: Full training loop with early stopping, AMP,
      GradNorm, checkpointing, and JSONL epoch logging.
    - _compute_class_weights: Inverse-frequency class weights.
"""
from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.checkpoints import load_checkpoint, load_training_state, save_checkpoint, save_training_state
from src.config import (
    CHECKPOINT_DIR,
    REPRO_ALLOW_BIG_CACHE,
    REPRO_ALLOW_UNC_WORKERS,
    REPRO_STRICT_BATCH_CHECKS,
    REPRO_TORCH_COMPILE_BACKEND,
)
from src.data import MultiTaskDataset, build_transforms, make_group_split
from src.loader_tuning import (
    _available_ram_gb,
    _initial_loader_tuning,
    _logical_cpu_count,
    _select_cache_size,
    resolve_batch_size,
)
from src.metrics import dice_coefficient
from src.models import GradNormBalancer, MultiTaskUNet
from src.utils import append_jsonl, fmt_seconds, now_iso

logger = logging.getLogger(__name__)


def _run_epoch(
    model,
    loader,
    optimizer,
    seg_criterion,
    cls_criterion,
    device,
    scaler,
    gradnorm: GradNormBalancer | None,
    seg_classes: int,
    train: bool,
    static_weights: bool = False,
    lambda_seg: float = 1.0,
    lambda_cls: float = 1.0,
    smoke_test: bool = False,
):
    """Run one training or validation epoch.

    Returns:
        (mean_loss, accuracy, mean_dice).
    """
    model.train() if train else model.eval()

    total_loss = 0.0
    total = 0
    correct = 0
    dice_vals: list[float] = []
    steps = 0
    use_amp = device == "cuda"
    shared_params = []
    if train:
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        shared_params = [p for p in base_model.unet.encoder.parameters() if p.requires_grad]

    with torch.set_grad_enabled(train):
        for batch_idx, (images, masks, labels) in enumerate(loader):
            if smoke_test and batch_idx >= 2:
                break

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if REPRO_STRICT_BATCH_CHECKS:
                if not torch.isfinite(images).all():
                    raise RuntimeError(f"Non-finite image values at batch {batch_idx}")
                if not torch.isfinite(masks).all():
                    raise RuntimeError(f"Non-finite mask values at batch {batch_idx}")
                if not torch.isfinite(labels.float()).all():
                    raise RuntimeError(f"Non-finite label values at batch {batch_idx}")

                labels_min = int(labels.min().item())
                labels_max = int(labels.max().item())
                cls_classes = None
                if hasattr(cls_criterion, "weight") and cls_criterion.weight is not None:
                    cls_classes = int(cls_criterion.weight.numel())
                if cls_classes is not None and (labels_min < 0 or labels_max >= cls_classes):
                    raise RuntimeError(
                        f"Classification label out of range at batch {batch_idx}: "
                        f"min={labels_min}, max={labels_max}, classes={cls_classes}"
                    )

                if seg_classes == 1:
                    masks_min = float(masks.min().item())
                    masks_max = float(masks.max().item())
                    if masks_min < -1e-6 or masks_max > 1.000001:
                        raise RuntimeError(
                            f"Binary mask out of range at batch {batch_idx}: "
                            f"min={masks_min}, max={masks_max}"
                        )
                else:
                    masks_min = int(masks.min().item())
                    masks_max = int(masks.max().item())
                    if masks_min < 0 or masks_max >= seg_classes:
                        raise RuntimeError(
                            f"Segmentation mask out of range at batch {batch_idx}: "
                            f"min={masks_min}, max={masks_max}, classes={seg_classes}"
                        )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                seg_out, cls_out = model(images)
                seg_loss = seg_criterion(seg_out, masks)
                cls_loss = cls_criterion(cls_out, labels)

                if static_weights:
                    loss = lambda_seg * seg_loss + lambda_cls * cls_loss
                elif gradnorm is not None:
                    w = gradnorm.weights()
                    loss = w[0].detach() * seg_loss + w[1].detach() * cls_loss
                else:
                    loss = seg_loss + cls_loss

            if train:
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                if gradnorm is not None and not static_weights and not bool(gradnorm.has_initial_losses.item()):
                    gradnorm.set_initial_losses(seg_loss.detach(), cls_loss.detach())

                gradnorm_loss = None
                if gradnorm is not None and not static_weights:
                    seg_grads = torch.autograd.grad(
                        seg_loss, shared_params, retain_graph=True, allow_unused=True
                    )
                    cls_grads = torch.autograd.grad(
                        cls_loss, shared_params, retain_graph=True, allow_unused=True
                    )

                    def _grad_norm(grads):
                        values = [g.norm() for g in grads if g is not None]
                        if not values:
                            return torch.tensor(0.0, device=device)
                        return torch.norm(torch.stack(values))

                    seg_norm = _grad_norm(list(seg_grads))
                    cls_norm = _grad_norm(list(cls_grads))
                    norms = torch.stack([seg_norm, cls_norm])

                    with torch.no_grad():
                        losses = torch.stack([seg_loss.detach(), cls_loss.detach()])
                        inv_rates = losses / (gradnorm.initial_losses + 1e-8)
                        inv_rates = inv_rates / inv_rates.mean().clamp(min=1e-8)
                        target = norms.detach().mean() * (inv_rates ** gradnorm.alpha)

                    weights = gradnorm.weights()
                    gradnorm_loss = torch.sum(
                        torch.abs(weights * norms.detach() - target)
                    )

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(retain_graph=gradnorm is not None and not static_weights)
                if gradnorm_loss is not None:
                    scaler.scale(gradnorm_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if gradnorm is not None and not static_weights:
                    gradnorm.normalize_()

            total_loss += float(loss.item()) if torch.isfinite(loss) else 0.0
            preds = cls_out.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
            dice_vals.append(dice_coefficient(seg_out.detach(), masks.detach(), seg_classes))
            steps += 1

    if steps == 0 or total == 0:
        return float("inf"), 0.0, 0.0

    return total_loss / steps, correct / total, float(np.mean(dice_vals))


def _make_collision_free_path(base_path: Path) -> Path:
    """Append a short timestamp suffix to prevent parallel-run collisions.

    Transforms ``dataset_encoder_best.pth`` into
    ``dataset_encoder_best_<YYYYMMDD><HHMMSS>.pth``.
    """
    stem = base_path.stem
    suffix = base_path.suffix
    ts = time.strftime("%Y%m%d%H%M%S")
    return base_path.with_name(f"{stem}_{ts}{suffix}")


def train_single_run(
    dataset: str,
    encoder: str,
    bundle: dict,
    meta: dict,
    args,
    device: str,
    epoch_log_file: Path,
    run_index: int,
    total_runs: int,
    skip_connections: bool = False,
    static_weights: bool = False,
    smoke_test: bool = False,
    run_label: str | None = None,
) -> dict:
    """Train a single (dataset, encoder) run.

    Args:
        dataset: Dataset key (tcga, panda, siim, pannuke).
        encoder: Encoder backbone name.
        bundle: Pre-loaded dataset bundle from ``load_dataset_bundle``.
        meta: Dataset metadata from ``DATASET_META``.
        args: Parsed CLI arguments (namespace).
        device: 'cuda' or 'cpu'.
        epoch_log_file: Path for JSONL epoch logging.
        run_index: 1-based index among total runs.
        total_runs: Total number of runs in the matrix.

    Returns:
        Dict with training metrics and metadata.
    """
    images = bundle["images"]
    masks = bundle["masks"]
    labels = bundle["labels"]
    groups = bundle["groups"]
    cpu_budget = _logical_cpu_count()
    available_ram_gb = _available_ram_gb()

    if len(images) < 100:
        raise RuntimeError(f"{dataset}: too few samples ({len(images)})")

    train_idx, val_idx = make_group_split(labels, groups, seed=args.seed, test_size=0.2)
    train_tf, val_tf = build_transforms(meta["img_size"])
    batch_size = resolve_batch_size(args.batch_size)
    attempt = 0
    binary_positive_min = int(meta.get("binary_positive_min", 1))
    crop_to_mask_bbox = bool(meta.get("crop_to_mask_bbox", False))

    requested_cache = int(getattr(args, "cache_size", -1))
    train_cache_size = requested_cache
    if dataset in {"panda", "siim", "pannuke"} and train_cache_size > 0 and not REPRO_ALLOW_BIG_CACHE:
        logger.info("%s: disabling dataset cache to avoid RAM OOM (cache_size was %d)", dataset.upper(), train_cache_size)
        train_cache_size = 0

    sample_paths = [str(p) for p in images[: min(len(images), 32)]]
    use_unc_paths = os.name == "nt" and any(p.startswith("\\\\wsl.localhost\\") for p in sample_paths)
    effective_workers, prefetch_factor, persistent_workers = _initial_loader_tuning(
        dataset, int(getattr(args, "num_workers", -1)), available_ram_gb, cpu_budget
    )
    if use_unc_paths and effective_workers > 0 and not REPRO_ALLOW_UNC_WORKERS:
        logger.info("Detected UNC/WSL dataset paths on Windows; forcing num_workers=0")
        effective_workers = 0
    elif use_unc_paths and effective_workers > 0 and REPRO_ALLOW_UNC_WORKERS:
        logger.info("UNC worker override enabled; using num_workers=%d", effective_workers)
    if effective_workers == 0:
        prefetch_factor = 0
        persistent_workers = False

    train_cache_size = _select_cache_size(dataset, requested_cache, available_ram_gb, effective_workers, REPRO_ALLOW_BIG_CACHE)

    while batch_size >= 2:
        attempt += 1
        start_ts = time.time()

        model = None
        optimizer = None
        scaler = None
        seg_criterion = None
        cls_criterion = None
        gradnorm = None
        train_loader = None
        val_loader = None
        train_ds = None
        val_ds = None

        train_ds = MultiTaskDataset(
            images[train_idx], masks[train_idx], labels[train_idx],
            seg_classes=meta["seg_classes"], binary_positive_min=binary_positive_min,
            crop_to_mask_bbox=crop_to_mask_bbox, transform=train_tf, cache_size=train_cache_size,
        )
        val_ds = MultiTaskDataset(
            images[val_idx], masks[val_idx], labels[val_idx],
            seg_classes=meta["seg_classes"], binary_positive_min=binary_positive_min,
            crop_to_mask_bbox=crop_to_mask_bbox, transform=val_tf, cache_size=0,
        )

        loader_kwargs = {
            "num_workers": effective_workers,
            "pin_memory": device == "cuda",
            "persistent_workers": (effective_workers > 0) and bool(persistent_workers),
        }
        if effective_workers > 0:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

        cls_w = _compute_class_weights(labels[train_idx], meta["num_classes"], device)

        model = MultiTaskUNet(
            encoder_name=encoder,
            num_classes=meta["num_classes"],
            seg_classes=meta["seg_classes"],
            skip_connections=not skip_connections,
        ).to(device)

        compile_active = False
        if args.compile and device == "cuda":
            try:
                if REPRO_TORCH_COMPILE_BACKEND:
                    model = torch.compile(model, backend=REPRO_TORCH_COMPILE_BACKEND)
                    logger.info("torch.compile enabled (backend=%s)", REPRO_TORCH_COMPILE_BACKEND)
                else:
                    model = torch.compile(model, mode="max-autotune")
                    logger.info("torch.compile enabled (mode=max-autotune)")
                compile_active = True
            except Exception as ex:
                logger.warning("torch.compile unavailable, continuing uncompiled: %s", ex)
                compile_active = False

        seg_criterion = nn.BCEWithLogitsLoss() if meta["seg_classes"] == 1 else nn.CrossEntropyLoss()
        cls_criterion = nn.CrossEntropyLoss(weight=cls_w)

        use_gradnorm = getattr(args, "use_gradnorm", False)
        gradnorm = None
        if use_gradnorm:
            gradnorm = GradNormBalancer(
                args.lambda_seg, args.lambda_cls, getattr(args, "gradnorm_alpha", 1.5)
            ).to(device)

        optimizer = optim.Adam(
            list(model.parameters()) + (list(gradnorm.parameters()) if gradnorm is not None else []),
            lr=args.lr,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_val_dice = 0.0
        patience_ctr = 0
        best_monitor_metric = float("inf")

        ckpt_base = args.checkpoint_dir / f"ckpt_{run_label}_best.pth"
        ckpt_path = _make_collision_free_path(ckpt_base)
        state_path = ckpt_path.with_suffix(".state.pt")
        start_epoch = 1
        resume_state = None

        logger.info(
            "[%d/%d] %s x %s | samples=%d train=%d val=%d bs=%d | "
            "workers=%d prefetch=%d persistent=%s cache=%d",
            run_index, total_runs, dataset.upper(), encoder,
            len(images), len(train_idx), len(val_idx), batch_size,
            effective_workers, prefetch_factor, persistent_workers, train_cache_size,
        )

        if args.resume and state_path.exists():
            resume_state = load_training_state(model, optimizer, gradnorm, ckpt_path, device)

        if resume_state is not None:
            start_epoch = int(resume_state.get("epoch", 0)) + 1
            best_val_loss = float(resume_state.get("best_val_loss", float("inf")))
            best_val_acc = float(resume_state.get("best_val_acc", 0.0))
            best_val_dice = float(resume_state.get("best_val_dice", 0.0))
            best_monitor_metric = float(resume_state.get("best_monitor_metric", best_val_loss))
            patience_ctr = int(resume_state.get("patience_ctr", 0))
            logger.info("Resuming from epoch %d using %s", start_epoch, state_path.name)

        logger.info("Ep | TrLoss TrAcc TrDice | VlLoss VlAcc VlDice | sec")

        effective_epochs = 1 if smoke_test else args.epochs
        if smoke_test:
            logger.info("SMOKE TEST MODE: epochs forced to 1, max 2 batches per epoch, checkpoint saving disabled")

        try:
            for epoch in range(start_epoch, effective_epochs + 1):
                t0 = time.time()
                tr_loss, tr_acc, tr_dice = _run_epoch(
                    model, train_loader, optimizer, seg_criterion, cls_criterion,
                    device, scaler, gradnorm, meta["seg_classes"], train=True,
                    static_weights=static_weights,
                    lambda_seg=args.lambda_seg,
                    lambda_cls=args.lambda_cls,
                    smoke_test=smoke_test,
                )
                vl_loss, vl_acc, vl_dice = _run_epoch(
                    model, val_loader, optimizer, seg_criterion, cls_criterion,
                    device, scaler, gradnorm, meta["seg_classes"], train=False,
                    static_weights=static_weights,
                    lambda_seg=args.lambda_seg,
                    lambda_cls=args.lambda_cls,
                    smoke_test=smoke_test,
                )
                ep_s = time.time() - t0

                is_best = vl_loss < best_monitor_metric
                if is_best:
                    best_val_loss = vl_loss
                    best_val_acc = vl_acc
                    best_val_dice = vl_dice
                    best_monitor_metric = vl_loss
                    patience_ctr = 0
                    if not smoke_test:
                        save_checkpoint(model, ckpt_path)
                    marker = "*"
                else:
                    patience_ctr += 1
                    marker = " "

                logger.info(
                    "  %2d%s | %6.4f %5.3f %6.3f | %6.4f %5.3f %6.3f | %4.1f",
                    epoch, marker, tr_loss, tr_acc, tr_dice,
                    vl_loss, vl_acc, vl_dice, ep_s,
                )

                append_jsonl(
                    epoch_log_file,
                    {
                        "timestamp": now_iso(),
                        "dataset": dataset,
                        "encoder": encoder,
                        "epoch": epoch,
                        "batch_size": batch_size,
                        "tr_loss": round(tr_loss, 6),
                        "tr_acc": round(tr_acc, 6),
                        "tr_dice": round(tr_dice, 6),
                        "vl_loss": round(vl_loss, 6),
                        "vl_acc": round(vl_acc, 6),
                        "vl_dice": round(vl_dice, 6),
                        "best_vl_loss": round(best_val_loss, 6),
                        "best_vl_acc": round(best_val_acc, 6),
                        "best_vl_dice": round(best_val_dice, 6),
                        "epoch_sec": round(ep_s, 2),
                        "is_best": bool(is_best),
                        "smoke_test": bool(smoke_test),
                    },
                )

                if not smoke_test:
                    save_training_state(
                        model, optimizer, gradnorm, ckpt_path, epoch,
                        best_val_loss, best_val_acc, best_val_dice,
                        best_monitor_metric, patience_ctr, batch_size,
                    )

                if patience_ctr >= args.patience:
                    logger.info("Early stop at epoch %d (patience=%d)", epoch, args.patience)
                    break

                if not (np.isfinite(tr_loss) and np.isfinite(vl_loss)):
                    logger.warning("NaN/Inf encountered, restoring best and ending run")
                    break

            if smoke_test:
                final_acc, final_dice = vl_acc, vl_dice
            else:
                load_checkpoint(model, ckpt_path, device)
                _, final_acc, final_dice = _run_epoch(
                    model, val_loader, optimizer, seg_criterion, cls_criterion,
                    device, scaler, gradnorm, meta["seg_classes"], train=False,
                    static_weights=static_weights,
                    lambda_seg=args.lambda_seg,
                    lambda_cls=args.lambda_cls,
                )

            duration = time.time() - start_ts
            return {
                "status": "completed",
                "dataset": dataset,
                "encoder": encoder,
                "samples": int(len(images)),
                "train_samples": int(len(train_idx)),
                "val_samples": int(len(val_idx)),
                "batch_size": int(batch_size),
                "loader_workers": int(effective_workers),
                "loader_prefetch_factor": int(prefetch_factor),
                "loader_persistent_workers": bool(persistent_workers),
                "loader_cache_size": int(train_cache_size),
                "available_ram_gb": float(round(available_ram_gb, 2)),
                "cpu_budget": int(cpu_budget),
                "best_val_loss": float(best_val_loss),
                "best_val_acc": float(best_val_acc),
                "best_val_dice": float(best_val_dice),
                "final_val_acc": float(final_acc),
                "final_val_dice": float(final_dice),
                "checkpoint": str(ckpt_path),
                "duration_sec": float(duration),
                "duration_hms": fmt_seconds(duration),
                "attempt": attempt,
                "compile_enabled": bool(compile_active),
                "compile_backend": REPRO_TORCH_COMPILE_BACKEND or None,
                "compile_mode": "max-autotune",
                "skip_connections_ablated": bool(skip_connections),
                "smoke_test": bool(smoke_test),
            }

        except RuntimeError as ex:
            msg = str(ex).lower()
            loader_failed = (
                "dataloader worker" in msg or "killed" in msg
                or "sigkill" in msg or "bus error" in msg or "broken pipe" in msg
            )
            if loader_failed and effective_workers > 0:
                new_workers = max(0, effective_workers // 2)
                logger.info(
                    "DataLoader instability detected. Retrying with num_workers %d -> %d",
                    effective_workers, new_workers,
                )
                effective_workers = new_workers
                prefetch_factor = 1
                persistent_workers = False
                train_cache_size = 0
                continue

            if "out of memory" in msg and device == "cuda":
                if int(getattr(args, "batch_size", 0) or 0) > 0:
                    if effective_workers > 0:
                        new_workers = max(0, effective_workers // 2)
                        logger.info("CUDA OOM: keeping batch_size=%d; reducing num_workers %d->%d", batch_size, effective_workers, new_workers)
                        effective_workers = new_workers
                        prefetch_factor = 1
                        persistent_workers = False
                        train_cache_size = 0
                        continue
                    raise

                if batch_size > 2:
                    logger.info("CUDA OOM at batch_size=%d; retrying with %d", batch_size, batch_size // 2)
                    batch_size = max(2, batch_size // 2)
                    continue

            raise
        finally:
            for obj in (model, optimizer, scaler, seg_criterion, cls_criterion, train_loader, val_loader, train_ds, val_ds):
                if obj is not None:
                    del obj
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as ex:
                    logger.warning("CUDA cache cleanup skipped: %s", ex)

    raise RuntimeError(f"Failed to train {dataset} x {encoder}: minimum batch size exhausted")


def _compute_class_weights(labels: np.ndarray, num_classes: int, device: str) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    if present.any():
        weights[present] = counts.sum() / (num_classes * counts[present])
    return torch.tensor(weights, dtype=torch.float32, device=device)