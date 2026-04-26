from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from repro.config import DEFAULT_BATCH_SIZE
    from repro.data import MultiTaskDataset, build_transforms, make_group_split
    from repro.utils import append_jsonl, fmt_seconds, now_iso
except ImportError:
    from .config import DEFAULT_BATCH_SIZE
    from .data import MultiTaskDataset, build_transforms, make_group_split
    from .utils import append_jsonl, fmt_seconds, now_iso


def _logical_cpu_count() -> int:
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return max(1, len(affinity))
    except Exception:
        pass
    return max(1, os.cpu_count() or 8)


def _available_ram_gb() -> float:
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
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_batch_size(requested: int | None) -> int:
    if requested and int(requested) > 0:
        return int(requested)
    return int(DEFAULT_BATCH_SIZE)


class MultiTaskUNet(nn.Module):
    def __init__(self, encoder_name: str, num_classes: int, seg_classes: int):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=seg_classes,
            activation=None,
        )

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
        decoder_out = self.unet.decoder(features)
        seg_out = self.unet.segmentation_head(decoder_out)
        cls_out = self.cls_head(bottleneck)
        return seg_out, cls_out


def dice_coefficient(seg_pred: torch.Tensor, seg_target: torch.Tensor, seg_classes: int) -> float:
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


def _run_epoch(
    model,
    loader,
    optimizer,
    seg_criterion,
    cls_criterion,
    device,
    scaler,
    lam_seg: float,
    lam_cls: float,
    seg_classes: int,
    train: bool,
):
    model.train() if train else model.eval()

    total_loss = 0.0
    total = 0
    correct = 0
    dice_vals: list[float] = []
    steps = 0
    strict_checks = _env_flag("REPRO_STRICT_BATCH_CHECKS", default=False)
    use_amp = device == "cuda"

    with torch.set_grad_enabled(train):
        for batch_idx, (images, masks, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if strict_checks:
                # Proactively fail on invalid data to avoid opaque CUDA kernel crashes.
                if not torch.isfinite(images).all():
                    raise RuntimeError(f"Non-finite image values at batch {batch_idx}")
                if not torch.isfinite(masks).all():
                    raise RuntimeError(f"Non-finite mask values at batch {batch_idx}")
                if not torch.isfinite(labels.float()).all():
                    raise RuntimeError(f"Non-finite label values at batch {batch_idx}")

                labels_min = int(labels.min().item())
                labels_max = int(labels.max().item())
                cls_classes = None
                # cls_classes isn't known until forward, so validate against criterion weights when present.
                if hasattr(cls_criterion, "weight") and cls_criterion.weight is not None:
                    cls_classes = int(cls_criterion.weight.numel())
                if cls_classes is not None and (labels_min < 0 or labels_max >= cls_classes):
                    raise RuntimeError(
                        f"Classification label out of range at batch {batch_idx}: min={labels_min}, max={labels_max}, classes={cls_classes}"
                    )

                if seg_classes == 1:
                    masks_min = float(masks.min().item())
                    masks_max = float(masks.max().item())
                    if masks_min < -1e-6 or masks_max > 1.000001:
                        raise RuntimeError(
                            f"Binary mask out of range at batch {batch_idx}: min={masks_min}, max={masks_max}"
                        )
                else:
                    masks_min = int(masks.min().item())
                    masks_max = int(masks.max().item())
                    if masks_min < 0 or masks_max >= seg_classes:
                        raise RuntimeError(
                            f"Segmentation mask out of range at batch {batch_idx}: min={masks_min}, max={masks_max}, classes={seg_classes}"
                        )

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                seg_out, cls_out = model(images)
                seg_loss = seg_criterion(seg_out, masks)
                cls_loss = cls_criterion(cls_out, labels)
                loss = lam_seg * seg_loss + lam_cls * cls_loss

            if train:
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += float(loss.item()) if torch.isfinite(loss) else 0.0
            preds = cls_out.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
            dice_vals.append(dice_coefficient(seg_out.detach(), masks.detach(), seg_classes))
            steps += 1

    if steps == 0 or total == 0:
        return float("inf"), 0.0, 0.0

    return total_loss / steps, correct / total, float(np.mean(dice_vals))


def _compute_class_weights(labels: np.ndarray, num_classes: int, device: str) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    if present.any():
        weights[present] = counts.sum() / (num_classes * counts[present])
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _select_cache_size(
    dataset: str,
    requested_cache: int,
    available_ram_gb: float,
    workers: int,
    allow_big_cache: bool,
) -> int:
    if requested_cache == 0:
        return 0

    if requested_cache > 0:
        if dataset in {"panda", "siim"} and not allow_big_cache:
            return 0
        return int(requested_cache)

    if dataset in {"tcga", "isic"} and available_ram_gb >= 18.0 and workers > 0:
        return 48 if dataset == "tcga" else 64

    return 0


def _initial_loader_tuning(
    dataset_name: str,
    requested_workers: int,
    available_ram_gb: float,
    cpu_budget: int,
) -> tuple[int, int, bool]:
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

        if dataset_name == "isic":
            target = 10 if available_ram_gb >= 20.0 else 8
            workers = min(cpu_budget, target)
            return workers, 4, True

        if dataset_name in {"panda", "siim"}:
            target = 10 if available_ram_gb >= 24.0 else 8
            workers = min(cpu_budget, target)
            return workers, 2, False

        workers = min(cpu_budget, 8)
        return workers, 2, True

    if dataset_name == "tcga":
        workers = min(requested_workers, min(cpu_budget, 12))
        return workers, 4, True

    if dataset_name == "isic":
        workers = min(requested_workers, min(cpu_budget, 10))
        return workers, 4, True

    if dataset_name in {"panda", "siim"}:
        workers = min(requested_workers, min(cpu_budget, 10))
        return workers, 2, False

    workers = min(requested_workers, min(cpu_budget, 8))
    return workers, 2, True


def _save_checkpoint(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save(state, path)


def _load_checkpoint(model, path: Path, device: str) -> None:
    state = torch.load(path, map_location=device)
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state)
    else:
        model.load_state_dict(state)


def _save_training_state(
    model,
    optimizer,
    path: Path,
    epoch: int,
    best_val_loss: float,
    best_val_acc: float,
    best_val_dice: float,
    best_monitor_metric: float,
    patience_ctr: int,
    batch_size: int,
) -> None:
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
        },
        path.with_suffix(".state.pt"),
    )


def _load_training_state(model, optimizer, path: Path, device: str) -> dict | None:
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

    return state


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
):
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

    # PANDA and SIIM samples can be large on disk; caching raw decoded arrays in RAM
    # (especially with multiple DataLoader workers + prefetch) can trigger host OOM kills
    # in WSL. Disabling cache does not change model math or hyperparameters.
    requested_cache = int(getattr(args, "cache_size", -1))
    train_cache_size = requested_cache
    allow_big_cache = _env_flag("REPRO_ALLOW_BIG_CACHE", default=False)
    if dataset in {"panda", "siim"} and train_cache_size > 0 and not allow_big_cache:
        print(f"  {dataset.upper()}: disabling dataset cache to avoid RAM OOM (cache_size was {train_cache_size})")
        train_cache_size = 0

    # Multiprocess workers can hang on Windows when reading from UNC WSL paths.
    sample_paths = [str(p) for p in images[: min(len(images), 32)]]
    use_unc_paths = os.name == "nt" and any(p.startswith("\\\\wsl.localhost\\") for p in sample_paths)
    allow_unc_workers = _env_flag("REPRO_ALLOW_UNC_WORKERS", default=False)
    effective_workers, prefetch_factor, persistent_workers = _initial_loader_tuning(
        dataset,
        int(getattr(args, "num_workers", -1)),
        available_ram_gb,
        cpu_budget,
    )
    if use_unc_paths and effective_workers > 0 and not allow_unc_workers:
        print("  Detected UNC/WSL dataset paths on Windows; forcing num_workers=0")
        effective_workers = 0
    elif use_unc_paths and effective_workers > 0 and allow_unc_workers:
        print(f"  UNC worker override enabled; using num_workers={effective_workers}")
    if effective_workers == 0:
        prefetch_factor = 0
        persistent_workers = False

    train_cache_size = _select_cache_size(
        dataset,
        requested_cache,
        available_ram_gb,
        effective_workers,
        allow_big_cache,
    )

    while batch_size >= 2:
        attempt += 1
        start_ts = time.time()

        model = None
        optimizer = None
        scaler = None
        seg_criterion = None
        cls_criterion = None
        train_loader = None
        val_loader = None
        train_ds = None
        val_ds = None

        train_ds = MultiTaskDataset(
            images[train_idx],
            masks[train_idx],
            labels[train_idx],
            seg_classes=meta["seg_classes"],
            binary_positive_min=binary_positive_min,
            crop_to_mask_bbox=crop_to_mask_bbox,
            transform=train_tf,
            cache_size=train_cache_size,
        )
        val_ds = MultiTaskDataset(
            images[val_idx],
            masks[val_idx],
            labels[val_idx],
            seg_classes=meta["seg_classes"],
            binary_positive_min=binary_positive_min,
            crop_to_mask_bbox=crop_to_mask_bbox,
            transform=val_tf,
            cache_size=0,
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
        ).to(device)
        
        compile_active = False

        if args.compile and device == "cuda":
            try:
                compile_backend = os.getenv("REPRO_TORCH_COMPILE_BACKEND", "").strip() or None
                compile_mode = os.getenv("REPRO_TORCH_COMPILE_MODE", "reduce-overhead").strip() or "reduce-overhead"

                if compile_backend:
                    model = torch.compile(model, backend=compile_backend)
                    print(f"  torch.compile enabled (backend={compile_backend})")
                else:
                    model = torch.compile(model, mode=compile_mode)
                    print(f"  torch.compile enabled (mode={compile_mode})")
                compile_active = True
            except Exception as ex:
                print(f"  torch.compile unavailable, continuing uncompiled: {ex}")
                compile_active = False

        seg_criterion = nn.BCEWithLogitsLoss() if meta["seg_classes"] == 1 else nn.CrossEntropyLoss()
        cls_criterion = nn.CrossEntropyLoss(weight=cls_w)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_val_dice = 0.0
        patience_ctr = 0

        # Use the joint objective (validation loss) for checkpoint selection.
        # This matches the paper's weighted multi-task loss optimization.
        best_monitor_metric = float("inf")

        ckpt_path = args.checkpoint_dir / f"{dataset}_{encoder}_best.pth"
        state_path = ckpt_path.with_suffix(".state.pt")
        start_epoch = 1
        resume_state = None

        print(
            f"[{run_index}/{total_runs}] {dataset.upper()} x {encoder} | "
            f"samples={len(images)} train={len(train_idx)} val={len(val_idx)} bs={batch_size} | "
            f"workers={effective_workers} prefetch={prefetch_factor} persistent={persistent_workers} cache={train_cache_size}"
        )
        if args.resume and state_path.exists():
            resume_state = _load_training_state(model, optimizer, ckpt_path, device)

        if resume_state is not None:
            start_epoch = int(resume_state.get("epoch", 0)) + 1
            best_val_loss = float(resume_state.get("best_val_loss", float("inf")))
            best_val_acc = float(resume_state.get("best_val_acc", 0.0))
            best_val_dice = float(resume_state.get("best_val_dice", 0.0))
            best_monitor_metric = float(resume_state.get("best_monitor_metric", best_val_loss))
            patience_ctr = int(resume_state.get("patience_ctr", 0))
            print(f"  Resuming from epoch {start_epoch} using {state_path.name}")

        print("  Ep | TrLoss TrAcc TrDice | VlLoss VlAcc VlDice | sec")

        try:
            for epoch in range(start_epoch, args.epochs + 1):
                t0 = time.time()
                tr_loss, tr_acc, tr_dice = _run_epoch(
                    model,
                    train_loader,
                    optimizer,
                    seg_criterion,
                    cls_criterion,
                    device,
                    scaler,
                    args.lambda_seg,
                    args.lambda_cls,
                    meta["seg_classes"],
                    train=True,
                )

                vl_loss, vl_acc, vl_dice = _run_epoch(
                    model,
                    val_loader,
                    optimizer,
                    seg_criterion,
                    cls_criterion,
                    device,
                    scaler,
                    args.lambda_seg,
                    args.lambda_cls,
                    meta["seg_classes"],
                    train=False,
                )

                ep_s = time.time() - t0
                
                # Keep the checkpoint with the best joint objective.
                is_best = vl_loss < best_monitor_metric

                if is_best:
                    best_val_loss = vl_loss
                    best_val_acc = vl_acc
                    best_val_dice = vl_dice
                    best_monitor_metric = vl_loss
                    patience_ctr = 0
                    _save_checkpoint(model, ckpt_path)
                    marker = "*"
                else:
                    patience_ctr += 1
                    marker = " "

                print(
                    f"  {epoch:>2}{marker} | {tr_loss:>6.4f} {tr_acc:>5.3f} {tr_dice:>6.3f} | "
                    f"{vl_loss:>6.4f} {vl_acc:>5.3f} {vl_dice:>6.3f} | {ep_s:>4.1f}"
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
                    },
                )

                _save_training_state(
                    model,
                    optimizer,
                    ckpt_path,
                    epoch,
                    best_val_loss,
                    best_val_acc,
                    best_val_dice,
                    best_monitor_metric,
                    patience_ctr,
                    batch_size,
                )

                if patience_ctr >= args.patience:
                    print(f"  Early stop at epoch {epoch} (patience={args.patience})")
                    break

                if not (np.isfinite(tr_loss) and np.isfinite(vl_loss)):
                    print("  NaN/Inf encountered, restoring best and ending run")
                    break

            _load_checkpoint(model, ckpt_path, device)
            _, final_acc, final_dice = _run_epoch(
                model,
                val_loader,
                optimizer,
                seg_criterion,
                cls_criterion,
                device,
                scaler,
                args.lambda_seg,
                args.lambda_cls,
                meta["seg_classes"],
                train=False,
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
                "compile_mode": os.getenv("REPRO_TORCH_COMPILE_MODE", "reduce-overhead").strip() or "reduce-overhead",
            }

        except RuntimeError as ex:
            msg = str(ex).lower()

            # Host-side loader failures (commonly from RAM pressure / OOM-killed workers).
            loader_failed = (
                "dataloader worker" in msg
                or "killed" in msg
                or "sigkill" in msg
                or "bus error" in msg
                or "broken pipe" in msg
            )
            if loader_failed and effective_workers > 0:
                new_workers = max(0, effective_workers // 2)
                print(
                    f"  DataLoader instability detected (likely host RAM pressure). "
                    f"Retrying with num_workers {effective_workers} -> {new_workers}, prefetch_factor -> 1, cache -> 0"
                )
                effective_workers = new_workers
                prefetch_factor = 1
                persistent_workers = False
                train_cache_size = 0
                continue

            # GPU OOM: keep paper batch size if explicitly requested; instead reduce input
            # pipeline RAM/worker pressure and retry. Only reduce batch size if auto-sized.
            if "out of memory" in msg and device == "cuda":
                if int(getattr(args, "batch_size", 0) or 0) > 0:
                    if effective_workers > 0:
                        new_workers = max(0, effective_workers // 2)
                        print(f"  CUDA OOM: keeping batch_size={batch_size}; reducing num_workers {effective_workers}->{new_workers}")
                        effective_workers = new_workers
                        prefetch_factor = 1
                        persistent_workers = False
                        train_cache_size = 0
                        continue
                    raise

                if batch_size > 2:
                    print(f"  CUDA OOM at batch_size={batch_size}; retrying with {batch_size // 2}")
                    batch_size = max(2, batch_size // 2)
                    continue

            raise
        finally:
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
            if scaler is not None:
                del scaler
            if seg_criterion is not None:
                del seg_criterion
            if cls_criterion is not None:
                del cls_criterion
            if train_loader is not None:
                del train_loader
            if val_loader is not None:
                del val_loader
            if train_ds is not None:
                del train_ds
            if val_ds is not None:
                del val_ds
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as ex:
                    # After a CUDA illegal-access failure the context can be poisoned;
                    # avoid masking the original error during teardown.
                    print(f"  CUDA cache cleanup skipped: {ex}")

    raise RuntimeError(f"Failed to train {dataset} x {encoder}: minimum batch size exhausted")
