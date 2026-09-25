#!/usr/bin/env python3
"""R3-1 NaN/Inf instrumented reproduce (1-epoch, per-batch non-finite capture).

Purpose
-------
The R3-1 gate (run_smoke_test.sh) logged ``TrLoss inf TrAcc 0.000 TrDice 0.000``
for epoch 1 of the TCGA x vgg16 2-epoch run, then soft-ended and crashed on a
missing ``best.pt``. This script reproduces ONE epoch of that exact run with
per-batch instrumentation so the root cause can be discriminated on the GPU:

  * per-batch: index, raw loss (seg/cls/total), LR, AMP/scaler state
    (dtype, scale, growth/backoff counters), grad-norm, and img/mask batch
    stats (min/max/any-NaN/any-Inf) -> written to a JSONL file.
  * A/B knobs to isolate the cause:
      --amp / --no-amp        toggle torch.autocast (bf16) on/off
      --compile / --no-compile  toggle torch.compile on/off
      --lr X                  override the learning rate
      --first-n-batches N      stop after N batches (fast triage)
      --batch-size B          override batch size
      --gradnorm / --no-gradnorm  toggle GradNorm (phase v2 default ON)
      --seed S                seed (default 42, matches the gate)

The script is CPU-safe (runs on CPU if no GPU) so it can be logic-verified
without a GPU, but it is intended to run GPU-side to reproduce the inf.

Usage (GPU-side, from the project root):
    python scripts/diag_nan.py --dataset tcga --encoder vgg16 \
        --amp --no-compile --lr 1e-4 --first-n-batches 8
    python scripts/diag_nan.py --dataset tcga --encoder vgg16 \
        --no-amp --no-compile --lr 1e-4 --first-n-batches 8
    python scripts/diag_nan.py --dataset tcga --encoder vgg16 \
        --amp --compile --lr 1e-4 --first-n-batches 8

Exit status: 0 = all batches finite; 1 = a non-finite batch was captured
(the JSONL + a printed diagnosis table are the deliverable).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# Make the project root importable when run as `python scripts/diag_nan.py`.
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import DATASET_META, DATASET_ROOTS
from src.data import MultiTaskDataset, build_transforms, load_dataset_bundle, make_group_split
from src.models import GradNormBalancer, MultiTaskUNet
from src.utils import now_iso

logger = logging.getLogger("diag_nan")


# ---------------------------------------------------------------------------
# Per-batch instrumentation
# ---------------------------------------------------------------------------
def _tensor_stats(t: torch.Tensor) -> dict:
    """min/max/any-NaN/any-Inf for a tensor (CPU-safe, no grad)."""
    if t is None:
        return {"min": None, "max": None, "any_nan": None, "any_inf": None}
    with torch.no_grad():
        tf = t.detach().float()
        return {
            "min": float(tf.min()),
            "max": float(tf.max()),
            "any_nan": bool(torch.isnan(tf).any()),
            "any_inf": bool(torch.isinf(tf).any()),
        }


def _scaler_state(scaler) -> dict:
    """AMP GradScaler state (dtype, scale, growth/backoff counters)."""
    if scaler is None:
        return {"enabled": False}
    out = {"enabled": bool(getattr(scaler, "enabled", False))}
    try:
        out["scale"] = float(scaler._scale.item())
    except Exception:
        out["scale"] = None
    out["growth_factor"] = getattr(scaler, "_growth_factor", None)
    out["growth_cnt"] = getattr(scaler, "_growth_cnt", None)
    out["backoff_cnt"] = getattr(scaler, "_backoff_cnt", None)
    return out


def _current_lr(optimizer) -> float:
    for pg in optimizer.param_groups:
        return float(pg["lr"])
    return float("nan")


def _is_finite(x) -> bool:
    if x is None:
        return True
    try:
        return bool(torch.isfinite(x))
    except Exception:
        return math.isfinite(float(x))


# ---------------------------------------------------------------------------
# One instrumented epoch
# ---------------------------------------------------------------------------
def run_instrumented_epoch(
    model,
    loader,
    optimizer,
    seg_criterion,
    cls_criterion,
    device,
    scaler,
    gradnorm,
    seg_classes: int,
    use_amp: bool,
    compile_active: bool,
    lambda_seg: float,
    lambda_cls: float,
    static_weights: bool,
    first_n_batches: int | None,
    out_path: Path,
    run_label: str,
    dataset: str,
    encoder: str,
    lr_override: float | None,
) -> dict:
    """Run one training epoch, capturing per-batch non-finite diagnostics.

    Returns a summary dict: {n_batches, n_nonfinite, first_nonfinite_batch,
    all_finite, out_path}.
    """
    model.train()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = out_path.open("w", encoding="utf-8")

    n_batches = 0
    n_nonfinite = 0
    first_nonfinite_batch = None
    last_grad_norm = None
    shared_params = []
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    shared_params = [p for p in base_model.unet.encoder.parameters() if p.requires_grad]

    amp_dtype = "bfloat16"
    logger.info(
        "Instrumented epoch: amp=%s (dtype=%s) compile=%s lr=%s gradnorm=%s "
        "static_weights=%s bs=%s first_n_batches=%s",
        use_amp, amp_dtype, compile_active, lr_override,
        gradnorm is not None, static_weights,
        getattr(loader, "batch_size", "?"), first_n_batches,
    )

    with torch.set_grad_enabled(True):
        for batch_idx, (images, masks, labels) in enumerate(loader):
            if first_n_batches is not None and batch_idx >= first_n_batches:
                break

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Data-batch stats (BEFORE the model) -> discriminates "bad data".
            img_stats = _tensor_stats(images)
            mask_stats = _tensor_stats(masks)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_amp and device == "cuda"):
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

            seg_loss_f = float(seg_loss.item())
            cls_loss_f = float(cls_loss.item())
            loss_f = float(loss.item())
            loss_finite = _is_finite(loss)
            seg_finite = _is_finite(seg_loss)
            cls_finite = _is_finite(cls_loss)

            # Backward + optimizer step (mirrors training.py) to capture the
            # real grad-norm and scaler state at this batch.
            if loss_finite:
                if gradnorm is not None and not static_weights \
                        and not bool(gradnorm.has_initial_losses.item()):
                    gradnorm.set_initial_losses(seg_loss.detach(), cls_loss.detach())

                gradnorm_loss = None
                if gradnorm is not None and not static_weights:
                    seg_grads = torch.autograd.grad(
                        seg_loss, shared_params, retain_graph=True, allow_unused=True)
                    cls_grads = torch.autograd.grad(
                        cls_loss, shared_params, retain_graph=True, allow_unused=True)

                    def _gnorm(grads):
                        vals = [g.norm() for g in grads if g is not None]
                        if not vals:
                            return torch.tensor(0.0, device=device)
                        return torch.norm(torch.stack(vals))

                    seg_norm = _gnorm(list(seg_grads))
                    cls_norm = _gnorm(list(cls_grads))
                    norms = torch.stack([seg_norm, cls_norm])
                    with torch.no_grad():
                        losses = torch.stack([seg_loss.detach(), cls_loss.detach()])
                        inv_rates = losses / (gradnorm.initial_losses + 1e-8)
                        inv_rates = inv_rates / inv_rates.mean().clamp(min=1e-8)
                        target = norms.detach().mean() * (inv_rates ** gradnorm.alpha)
                    weights = gradnorm.weights()
                    gradnorm_loss = torch.sum(
                        torch.abs(weights * norms.detach() - target))

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(
                    retain_graph=gradnorm is not None and not static_weights)
                if gradnorm_loss is not None:
                    scaler.scale(gradnorm_loss).backward()
                clip_params = list(model.parameters())
                if gradnorm is not None and not static_weights:
                    clip_params += list(gradnorm.parameters())
                last_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0).item())
                scaler.step(optimizer)
                scaler.update()
                if gradnorm is not None and not static_weights:
                    gradnorm.normalize_()
            else:
                optimizer.zero_grad(set_to_none=True)

            n_batches += 1
            record = {
                "run_label": run_label,
                "dataset": dataset,
                "encoder": encoder,
                "batch_idx": int(batch_idx),
                "batch_size": int(images.size(0)),
                "seg_loss": seg_loss_f,
                "cls_loss": cls_loss_f,
                "total_loss": loss_f,
                "seg_finite": bool(seg_finite),
                "cls_finite": bool(cls_finite),
                "total_finite": bool(loss_finite),
                "lr": _current_lr(optimizer),
                "amp": {"enabled": bool(use_amp and device == "cuda"),
                        "dtype": amp_dtype},
                "scaler": _scaler_state(scaler),
                "grad_norm": last_grad_norm,
                "img": img_stats,
                "mask": mask_stats,
                "compile_active": bool(compile_active),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "timestamp": now_iso(),
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            status = "OK " if loss_finite else "NON-FINITE"
            logger.info(
                "  batch %3d [%s] seg=%.4g cls=%.4g total=%.4g lr=%.3g "
                "scale=%s grad_norm=%s img[min=%s max=%s nan=%s] "
                "mask[min=%s max=%s nan=%s]",
                batch_idx, status, seg_loss_f, cls_loss_f, loss_f,
                _current_lr(optimizer),
                _scaler_state(scaler).get("scale"),
                last_grad_norm,
                img_stats["min"], img_stats["max"], img_stats["any_nan"],
                mask_stats["min"], mask_stats["max"], mask_stats["any_nan"],
            )

            if not loss_finite:
                n_nonfinite += 1
                if first_nonfinite_batch is None:
                    first_nonfinite_batch = batch_idx
                    logger.error(
                        "  >>> FIRST NON-FINITE BATCH = %d (seg_finite=%s "
                        "cls_finite=%s). Full record in %s",
                        batch_idx, seg_finite, cls_finite, out_path,
                    )
                    # Keep going a few more batches to see if it is
                    # persistent (data) or transient (AMP/scaler), but stop
                    # early if the caller asked for a small window.
                    if first_n_batches is not None and batch_idx + 1 >= first_n_batches:
                        break

    out_f.close()
    summary = {
        "n_batches": n_batches,
        "n_nonfinite": n_nonfinite,
        "first_nonfinite_batch": first_nonfinite_batch,
        "all_finite": n_nonfinite == 0,
        "out_path": str(out_path),
    }
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="R3-1 NaN/Inf instrumented 1-epoch reproduce (per-batch capture).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="tcga",
                   choices=["tcga", "panda", "siim", "pannuke"])
    p.add_argument("--encoder", default="vgg16",
                   choices=["vgg16", "mobilenet_v2"])
    p.add_argument("--lr", type=float, default=1e-4,
                   help="learning rate (phase v2 default 1e-4)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--first-n-batches", type=int, default=None,
                   help="stop after N batches (fast triage); default = full epoch")
    p.add_argument("--epochs", type=int, default=1,
                   help="number of epochs (default 1; the gate's failing epoch)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--run-label", default=None,
                   help="output label (default: diag_<dataset>_<encoder>)")
    p.add_argument("--out-dir", default=None,
                   help="output dir (default: results/kfold_campaign/<run-label>/diag)")

    amp_g = p.add_mutually_exclusive_group()
    amp_g.add_argument("--amp", dest="amp", action="store_true", default=None,
                       help="enable bf16 autocast (default: on for CUDA)")
    amp_g.add_argument("--no-amp", dest="amp", action="store_false",
                       help="disable bf16 autocast")

    comp_g = p.add_mutually_exclusive_group()
    comp_g.add_argument("--compile", dest="compile", action="store_true",
                        default=None, help="enable torch.compile")
    comp_g.add_argument("--no-compile", dest="compile", action="store_false",
                        help="disable torch.compile (default)")

    gn_g = p.add_mutually_exclusive_group()
    gn_g.add_argument("--gradnorm", dest="gradnorm", action="store_true",
                      default=None, help="enable GradNorm (phase v2 default)")
    gn_g.add_argument("--no-gradnorm", dest="gradnorm", action="store_false",
                      help="disable GradNorm")

    p.add_argument("--static-weights", action="store_true",
                   help="use fixed lambda weights (no GradNorm)")
    p.add_argument("--lambda-seg", type=float, default=5.0)
    p.add_argument("--lambda-cls", type=float, default=1.0)
    return p


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    args = build_parser().parse_args()

    # Resolve the A/B knobs (default: AMP on for CUDA, compile off, gradnorm on
    # to mirror the phase-v2 gate run).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (args.amp if args.amp is not None else (device == "cuda"))
    compile_on = (args.compile if args.compile is not None else False)
    use_gradnorm = (args.gradnorm if args.gradnorm is not None else True)
    if args.static_weights:
        use_gradnorm = False

    run_label = args.run_label or f"diag_{args.dataset}_{args.encoder}"
    out_dir = Path(args.out_dir) if args.out_dir else (
        BASE_DIR / "results" / "kfold_campaign" / run_label / "diag")
    out_path = out_dir / "per_batch.jsonl"

    logger.info("Device: %s | amp=%s compile=%s gradnorm=%s lr=%g bs=%d "
                "first_n_batches=%s seed=%d",
                device, use_amp, compile_on, use_gradnorm, args.lr,
                args.batch_size, args.first_n_batches, args.seed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    meta = DATASET_META[args.dataset]
    bundle = load_dataset_bundle(
        args.dataset, DATASET_ROOTS[args.dataset], skip_macenko=False)
    images, masks, labels, groups = (
        bundle["images"], bundle["masks"], bundle["labels"], bundle["groups"])
    logger.info("%s: samples=%d groups=%d", args.dataset.upper(),
                len(images), len(np.unique(groups)))

    train_idx, val_idx = make_group_split(
        labels, groups, seed=args.seed, test_size=0.2,
        grouping=meta.get("grouping"),
        provenance_path=DATASET_ROOTS[args.dataset] / "preprocessed" /
        "grouping_provenance.json",
    )
    train_tf, _ = build_transforms(meta["img_size"])

    train_ds = MultiTaskDataset(
        images[train_idx], masks[train_idx], labels[train_idx],
        seg_classes=meta["seg_classes"],
        binary_positive_min=int(meta.get("binary_positive_min", 1)),
        crop_to_mask_bbox=bool(meta.get("crop_to_mask_bbox", False)),
        transform=train_tf, cache_size=0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=False,
    )

    model = MultiTaskUNet(
        encoder_name=args.encoder,
        num_classes=meta["num_classes"],
        seg_classes=meta["seg_classes"],
        skip_connections=True,
    ).to(device)

    compile_active = False
    if compile_on and device == "cuda":
        try:
            model = torch.compile(model, mode="max-autotune")
            compile_active = True
            logger.info("torch.compile enabled (mode=max-autotune)")
        except Exception as ex:
            logger.warning("torch.compile unavailable, continuing uncompiled: %s", ex)

    seg_criterion = nn.BCEWithLogitsLoss() if meta["seg_classes"] == 1 else nn.CrossEntropyLoss()
    cls_criterion = nn.CrossEntropyLoss()

    gradnorm = None
    if use_gradnorm:
        gradnorm = GradNormBalancer(args.lambda_seg, args.lambda_cls, 1.5).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + (list(gradnorm.parameters()) if gradnorm is not None else []),
        lr=args.lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and use_amp))

    overall = {"n_batches": 0, "n_nonfinite": 0, "first_nonfinite_batch": None,
               "all_finite": True, "epochs": []}
    for epoch in range(1, args.epochs + 1):
        logger.info("=== Epoch %d/%d ===", epoch, args.epochs)
        ep_summary = run_instrumented_epoch(
            model, train_loader, optimizer, seg_criterion, cls_criterion,
            device, scaler, gradnorm, meta["seg_classes"],
            use_amp=use_amp, compile_active=compile_active,
            lambda_seg=args.lambda_seg, lambda_cls=args.lambda_cls,
            static_weights=args.static_weights,
            first_n_batches=args.first_n_batches,
            out_path=out_path, run_label=run_label,
            dataset=args.dataset, encoder=args.encoder,
            lr_override=args.lr,
        )
        overall["n_batches"] += ep_summary["n_batches"]
        overall["n_nonfinite"] += ep_summary["n_nonfinite"]
        if ep_summary["first_nonfinite_batch"] is not None and \
                overall["first_nonfinite_batch"] is None:
            overall["first_nonfinite_batch"] = ep_summary["first_nonfinite_batch"]
        overall["all_finite"] = overall["all_finite"] and ep_summary["all_finite"]
        overall["epochs"].append(ep_summary)

    # Final verdict
    logger.info("=" * 64)
    if overall["all_finite"]:
        logger.info("VERDICT: all %d batches FINITE (no NaN/Inf reproduced).",
                    overall["n_batches"])
        logger.info("  -> the gate's inf was NOT reproduced under this config "
                    "(amp=%s compile=%s gradnorm=%s lr=%g).",
                    use_amp, compile_on, use_gradnorm, args.lr)
    else:
        logger.info("VERDICT: %d NON-FINITE batch(es); first at batch %s.",
                    overall["n_nonfinite"], overall["first_nonfinite_batch"])
        logger.info("  -> reproduced under amp=%s compile=%s gradnorm=%s lr=%g.",
                    use_amp, compile_on, use_gradnorm, args.lr)
    logger.info("Per-batch JSONL: %s", out_path)
    logger.info("=" * 64)

    # Write a small machine-readable verdict next to the JSONL.
    verdict_path = out_dir / "verdict.json"
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    with verdict_path.open("w", encoding="utf-8") as vf:
        json.dump({
            "run_label": run_label,
            "dataset": args.dataset,
            "encoder": args.encoder,
            "device": device,
            "amp": use_amp,
            "compile": compile_on,
            "gradnorm": use_gradnorm,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "first_n_batches": args.first_n_batches,
            "n_batches": overall["n_batches"],
            "n_nonfinite": overall["n_nonfinite"],
            "first_nonfinite_batch": overall["first_nonfinite_batch"],
            "all_finite": overall["all_finite"],
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "timestamp": now_iso(),
        }, vf, indent=2)
    logger.info("Verdict JSON: %s", verdict_path)

    return 0 if overall["all_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
