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
    from repro.data import MultiTaskDataset, build_transforms, make_group_split
    from repro.utils import append_jsonl, fmt_seconds, now_iso
except ImportError:
    from .data import MultiTaskDataset, build_transforms, make_group_split
    from .utils import append_jsonl, fmt_seconds, now_iso

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def estimate_batch_size(dataset: str, encoder: str, img_size: int, requested: int | None) -> int:
    if requested and requested > 0:
        return int(requested)

    if encoder == "mobilenet_v2":
        if img_size <= 128:
            return 48
        if img_size <= 224:
            return 32
        return 24

    if img_size <= 128:
        return 36
    if img_size <= 224:
        return 20
    return 12


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
        intersection = (pred * seg_target).sum()
        union = pred.sum() + seg_target.sum()
        if float(union) == 0.0:
            return 1.0
        return float((2.0 * intersection) / (union + 1e-8))

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
    use_amp = device == "cuda"

    with torch.set_grad_enabled(train):
        for images, masks, labels in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

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


def _save_checkpoint(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save(state, path)


def _load_checkpoint(model, path: Path, device: str) -> None:
    state = torch.load(path, map_location=device, weights_only=True)
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state)
    else:
        model.load_state_dict(state)


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

    if len(images) < 100:
        raise RuntimeError(f"{dataset}: too few samples ({len(images)})")

    train_idx, val_idx = make_group_split(labels, groups, seed=args.seed, test_size=0.2)
    train_tf, val_tf = build_transforms(meta["img_size"])

    batch_size = estimate_batch_size(dataset, encoder, meta["img_size"], args.batch_size)
    attempt = 0

    # Multiprocess workers can hang on Windows when reading from UNC WSL paths.
    sample_paths = [str(p) for p in images[: min(len(images), 32)]]
    use_unc_paths = os.name == "nt" and any(p.startswith("\\\\wsl.localhost\\") for p in sample_paths)
    allow_unc_workers = _env_flag("REPRO_ALLOW_UNC_WORKERS", default=False)
    effective_workers = int(args.num_workers)
    if use_unc_paths and effective_workers > 0 and not allow_unc_workers:
        print("  Detected UNC/WSL dataset paths on Windows; forcing num_workers=0")
        effective_workers = 0
    elif use_unc_paths and effective_workers > 0 and allow_unc_workers:
        print(f"  UNC worker override enabled; using num_workers={effective_workers}")

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
            transform=train_tf,
            cache_size=args.cache_size,
        )
        val_ds = MultiTaskDataset(
            images[val_idx],
            masks[val_idx],
            labels[val_idx],
            seg_classes=meta["seg_classes"],
            transform=val_tf,
            cache_size=0,
        )

        loader_kwargs = {
            "num_workers": effective_workers,
            "pin_memory": device == "cuda",
            "persistent_workers": effective_workers > 0,
        }
        if effective_workers > 0:
            loader_kwargs["prefetch_factor"] = 4

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

        cls_w = _compute_class_weights(labels[train_idx], meta["num_classes"], device)

        model = MultiTaskUNet(
            encoder_name=encoder,
            num_classes=meta["num_classes"],
            seg_classes=meta["seg_classes"],
        ).to(device)

        if args.compile and device == "cuda":
            try:
                model = torch.compile(model)
                print("  torch.compile enabled")
            except Exception as ex:
                print(f"  torch.compile unavailable, continuing uncompiled: {ex}")

        seg_criterion = nn.BCEWithLogitsLoss() if meta["seg_classes"] == 1 else nn.CrossEntropyLoss()
        cls_criterion = nn.CrossEntropyLoss(weight=cls_w)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_val_dice = 0.0
        patience_ctr = 0

        ckpt_path = args.checkpoint_dir / f"{dataset}_{encoder}_best.pth"

        print(
            f"[{run_index}/{total_runs}] {dataset.upper()} x {encoder} | "
            f"samples={len(images)} train={len(train_idx)} val={len(val_idx)} bs={batch_size}"
        )
        print("  Ep | TrLoss TrAcc TrDice | VlLoss VlAcc VlDice | sec")

        try:
            for epoch in range(1, args.epochs + 1):
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
                is_best = vl_loss < best_val_loss

                if is_best:
                    best_val_loss = vl_loss
                    best_val_acc = vl_acc
                    best_val_dice = vl_dice
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
                "best_val_loss": float(best_val_loss),
                "best_val_acc": float(best_val_acc),
                "best_val_dice": float(best_val_dice),
                "final_val_acc": float(final_acc),
                "final_val_dice": float(final_dice),
                "checkpoint": str(ckpt_path),
                "duration_sec": float(duration),
                "duration_hms": fmt_seconds(duration),
                "attempt": attempt,
            }

        except RuntimeError as ex:
            if "out of memory" in str(ex).lower() and device == "cuda" and batch_size > 2:
                print(f"  OOM at batch_size={batch_size}; retrying with {batch_size // 2}")
                batch_size = max(2, batch_size // 2)
            else:
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
                torch.cuda.empty_cache()

    raise RuntimeError(f"Failed to train {dataset} x {encoder}: minimum batch size exhausted")
