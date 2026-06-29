"""Multi-task UNet architecture and GradNorm loss balancer.

Provides:
    - MultiTaskUNet: Hard parameter-sharing UNet with shared encoder,
      separate segmentation head (via SMP decoder) and classification head.
    - GradNormBalancer: Dynamic multi-task loss weighting following
      Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss
      Balancing in Multi-task Learning", ICML 2018.
    - train_single_run: Full training loop with early stopping, AMP,
      GradNorm, checkpointing, and JSONL epoch logging.
"""
from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import DEFAULT_BATCH_SIZE
from src.data import MultiTaskDataset, build_transforms, make_group_split
from src.utils import append_jsonl, fmt_seconds, now_iso

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System helpers
# ---------------------------------------------------------------------------

def _logical_cpu_count() -> int:
    """Return the number of logical CPUs available to this process."""
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return max(1, len(affinity))
    except Exception:
        pass
    return max(1, os.cpu_count() or 8)


def _available_ram_gb() -> float:
    """Return available RAM in GB (Linux /proc/meminfo)."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return 0.0
    available_kb = None
    total_kb = None
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available_kb = float(line.split()[1])
            elif line.startswith("MemTotal:"):
                total_kb = float(line.split()[1])
        value_kb = available_kb if available_kb is not None else total_kb
        if value_kb is None:
            return 0.0
        return float(value_kb) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

# ---------------------------------------------------------------------------
# GradNorm
# ---------------------------------------------------------------------------

class GradNormBalancer(nn.Module):
    """GradNorm dynamic loss balancer for two tasks.

    Maintains learnable log-weights for segmentation and classification losses.
    After each batch, gradient magnitudes on shared parameters are balanced
    so that neither task dominates training.

    Args:
        init_seg: Initial segmentation loss weight.
        init_cls: Initial classification loss weight.
        alpha: GradNorm asymmetry exponent (default 1.5).
    """

    def __init__(self, init_seg: float, init_cls: float, alpha: float):
        super().__init__()
        init = torch.tensor([float(init_seg), float(init_cls)], dtype=torch.float32)
        self.log_weights = nn.Parameter(torch.log(init.clamp(min=1e-4)))
        self.alpha = float(alpha)
        self.register_buffer("initial_losses", torch.zeros(2, dtype=torch.float32))
        self.register_buffer("has_initial_losses", torch.tensor(False))

    def weights(self) -> torch.Tensor:
        """Return current (positive) loss weights."""
        return torch.exp(self.log_weights)

    def normalize_(self) -> None:
        """Normalize weights so they sum to 2."""
        with torch.no_grad():
            w = self.weights().clamp(min=1e-4)
            w = w * (2.0 / w.sum().clamp(min=1e-4))
            self.log_weights.copy_(w.log())

    def set_initial_losses(self, seg_loss: torch.Tensor, cls_loss: torch.Tensor) -> None:
        """Record initial per-task losses for relative-rate computation."""
        with torch.no_grad():
            self.initial_losses.copy_(
                torch.tensor([float(seg_loss), float(cls_loss)], device=self.initial_losses.device)
            )
            self.has_initial_losses.fill_(True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiTaskUNet(nn.Module):
    """Multi-task U-Net with hard parameter sharing.

    Uses a shared SMP Unet encoder for feature extraction. The decoder +
    segmentation head produces pixel-wise masks, while a lightweight
    classification head (global-pool → MLP) produces class logits.

    Args:
        encoder_name: SMP encoder backbone (e.g., 'vgg16', 'mobilenet_v2').
        num_classes: Number of classification categories.
        seg_classes: Number of segmentation classes (1 = binary sigmoid).
        skip_connections: If False, bypass UNet decoder skip connections
            by zeroing intermediate encoder features (ablation study).
    """

    def __init__(
        self,
        encoder_name: str,
        num_classes: int,
        seg_classes: int,
        skip_connections: bool = True,
    ):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=seg_classes,
            activation=None,
        )
        self._skip_connections = skip_connections
        bottleneck_dim = self.unet.encoder.out_channels[-1]
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.unet.encoder(x)
        bottleneck = features[-1]

        # Ablation: zero out skip connections (all encoder features except
        # the bottleneck) to test the necessity of high-resolution feature
        # transfer in the UNet decoder.
        if not self._skip_connections:
            features = [
                torch.zeros_like(f) for f in features[:-1]
            ] + [bottleneck]

        decoder_out = self.unet.decoder(features)
        seg_out = self.unet.segmentation_head(decoder_out)
        cls_out = self.cls_head(bottleneck)
        return seg_out, cls_out

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Training loop internals
# ---------------------------------------------------------------------------

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
    strict_checks = _env_flag("REPRO_STRICT_BATCH_CHECKS", default=False)
    use_amp = device == "cuda"
    shared_params = []
    if train:
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        shared_params = [p for p in base_model.unet.encoder.parameters() if p.requires_grad]

    with torch.set_grad_enabled(train):
        for batch_idx, (images, masks, labels) in enumerate(loader):
            # Smoke test: stop after exactly 2 batches
            if smoke_test and batch_idx >= 2:
                break

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if strict_checks:
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

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(model, path: Path) -> None:
    """Save model state_dict to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save(state, path)


def _load_checkpoint(model, path: Path, device: str) -> None:
    """Load model state_dict from *path*."""
    state = torch.load(path, map_location=device)
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state)
    else:
        model.load_state_dict(state)


def _save_training_state(
    model, optimizer, gradnorm, path: Path, epoch: int,
    best_val_loss: float, best_val_acc: float, best_val_dice: float,
    best_monitor_metric: float, patience_ctr: int, batch_size: int,
) -> None:
    """Save full training state for resuming."""
    model_state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save(
        {
            "epoch": int(epoch),
            "batch_size": int(batch_size),
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": float(best_val_loss),
            "best_val_acc": float(best_val_acc),
            "best_val_dice": float(best_val_dice),
            "best_monitor_metric": float(best_monitor_metric),
            "patience_ctr": int(patience_ctr),
            "gradnorm_log_weights": gradnorm.log_weights.detach().cpu() if gradnorm is not None else None,
            "gradnorm_initial_losses": gradnorm.initial_losses.detach().cpu() if gradnorm is not None else None,
        },
        path.with_suffix(".state.pt"),
    )


def _load_training_state(model, optimizer, gradnorm, path: Path, device: str) -> dict | None:
    """Resume training state from checkpoint. Returns state dict or None."""
    state_path = path.with_suffix(".state.pt")
    if not state_path.exists():
        return None
    state = torch.load(state_path, map_location=device)
    model_state = state.get("model_state")
    if model_state is None:
        return None
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)
    optimizer_state = state.get("optimizer_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if gradnorm is not None:
        log_weights = state.get("gradnorm_log_weights")
        if log_weights is not None:
            gradnorm.log_weights.data.copy_(log_weights.to(device))
        initial_losses = state.get("gradnorm_initial_losses")
        if initial_losses is not None:
            gradnorm.initial_losses.data.copy_(initial_losses.to(device))
            gradnorm.has_initial_losses.fill_(True)
    return state

# ---------------------------------------------------------------------------
# DataLoader tuning
# ---------------------------------------------------------------------------

def resolve_batch_size(requested: int | None) -> int:
    if requested and int(requested) > 0:
        return int(requested)
    return int(DEFAULT_BATCH_SIZE)


def _select_cache_size(dataset: str, requested_cache: int, available_ram_gb: float, workers: int, allow_big_cache: bool) -> int:
    if requested_cache == 0:
        return 0
    if requested_cache > 0:
        if dataset in {"panda", "siim", "pannuke"} and not allow_big_cache:
            return 0
        return int(requested_cache)
    if dataset == "tcga" and available_ram_gb >= 18.0 and workers > 0:
        return 48 if dataset == "tcga" else 64
    return 0


def _initial_loader_tuning(dataset_name: str, requested_workers: int, available_ram_gb: float, cpu_budget: int):
    """Return (num_workers, prefetch_factor, persistent_workers)."""
    requested_workers = int(requested_workers)
    cpu_budget = max(1, int(cpu_budget))
    if requested_workers == 0:
        return 0, 0, False
    if requested_workers < 0:
        if dataset_name == "tcga":
            target = 12 if available_ram_gb >= 20.0 else 8
            workers = min(cpu_budget, target)
            return workers, 4, True
        if dataset_name in {"panda", "siim", "pannuke"}:
            target = 10 if available_ram_gb >= 24.0 else 8
            workers = min(cpu_budget, target)
            return workers, 2, False
        workers = min(cpu_budget, 8)
        return workers, 2, True
    if dataset_name == "tcga":
        workers = min(requested_workers, min(cpu_budget, 12))
        return workers, 4, True
    if dataset_name in {"panda", "siim", "pannuke"}:
        workers = min(requested_workers, min(cpu_budget, 10))
        return workers, 2, False
    workers = min(requested_workers, min(cpu_budget, 8))
    return workers, 2, True

# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------

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
    allow_big_cache = _env_flag("REPRO_ALLOW_BIG_CACHE", default=False)
    if dataset in {"panda", "siim", "pannuke"} and train_cache_size > 0 and not allow_big_cache:
        logger.info("%s: disabling dataset cache to avoid RAM OOM (cache_size was %d)", dataset.upper(), train_cache_size)
        train_cache_size = 0

    sample_paths = [str(p) for p in images[: min(len(images), 32)]]
    use_unc_paths = os.name == "nt" and any(p.startswith("\\\\wsl.localhost\\") for p in sample_paths)
    allow_unc_workers = _env_flag("REPRO_ALLOW_UNC_WORKERS", default=False)
    effective_workers, prefetch_factor, persistent_workers = _initial_loader_tuning(
        dataset, int(getattr(args, "num_workers", -1)), available_ram_gb, cpu_budget
    )
    if use_unc_paths and effective_workers > 0 and not allow_unc_workers:
        logger.info("Detected UNC/WSL dataset paths on Windows; forcing num_workers=0")
        effective_workers = 0
    elif use_unc_paths and effective_workers > 0 and allow_unc_workers:
        logger.info("UNC worker override enabled; using num_workers=%d", effective_workers)
    if effective_workers == 0:
        prefetch_factor = 0
        persistent_workers = False

    train_cache_size = _select_cache_size(dataset, requested_cache, available_ram_gb, effective_workers, allow_big_cache)

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
                # max-autotune is optimized for large-batch training on Ampere+
                # and performs extensive autotuning for maximum throughput.
                compile_backend = os.getenv("REPRO_TORCH_COMPILE_BACKEND", "").strip() or None
                if compile_backend:
                    model = torch.compile(model, backend=compile_backend)
                    logger.info("torch.compile enabled (backend=%s)", compile_backend)
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

        ckpt_path = args.checkpoint_dir / f"{dataset}_{encoder}_best.pth"
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
            resume_state = _load_training_state(model, optimizer, gradnorm, ckpt_path, device)

        if resume_state is not None:
            start_epoch = int(resume_state.get("epoch", 0)) + 1
            best_val_loss = float(resume_state.get("best_val_loss", float("inf")))
            best_val_acc = float(resume_state.get("best_val_acc", 0.0))
            best_val_dice = float(resume_state.get("best_val_dice", 0.0))
            best_monitor_metric = float(resume_state.get("best_monitor_metric", best_val_loss))
            patience_ctr = int(resume_state.get("patience_ctr", 0))
            logger.info("Resuming from epoch %d using %s", start_epoch, state_path.name)

        logger.info("Ep | TrLoss TrAcc TrDice | VlLoss VlAcc VlDice | sec")

        # Smoke test: force exactly 1 epoch
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
                    # Smoke test: skip checkpoint saving to avoid overwriting real weights
                    if not smoke_test:
                        _save_checkpoint(model, ckpt_path)
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

                # Smoke test: skip training state saving
                if not smoke_test:
                    _save_training_state(
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

            # Smoke test: skip final checkpoint restore; reuse last epoch val metrics
            if smoke_test:
                final_acc, final_dice = vl_acc, vl_dice
            else:
                _load_checkpoint(model, ckpt_path, device)
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
                "compile_backend": os.getenv("REPRO_TORCH_COMPILE_BACKEND", "").strip() or None,
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