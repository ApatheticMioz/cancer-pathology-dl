"""Epoch loops and training orchestrator.

Provides:
    - _run_epoch: Single training or validation epoch.
    - train_single_run: Full training loop with early stopping, AMP,
      GradNorm, checkpointing, and JSONL epoch logging.
    - _compute_class_weights: Inverse-frequency class weights.
"""
from __future__ import annotations

import gc
import json
import logging
import math
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
    DATASET_ROOTS,
    GRAD_CLIP_MAX_NORM,
    GRAD_SKIP_MAX_PER_FOLD,
    GRADNORM_WEIGHT_CLAMP,
    REPRO_ALLOW_BIG_CACHE,
    REPRO_ALLOW_UNC_WORKERS,
    REPRO_STRICT_BATCH_CHECKS,
    REPRO_TORCH_COMPILE_BACKEND,
    RESULTS_DIR,
)
from src.data import MultiTaskDataset, build_transforms, make_group_kfold_splits, make_group_split
from src.loader_tuning import (
    _available_ram_gb,
    _initial_loader_tuning,
    _logical_cpu_count,
    _select_cache_size,
    resolve_batch_size,
)
from src.metrics import dice_coefficient, dice_coefficient_per_sample, iou_coefficient, positive_slice_dice
from src.models import GradNormBalancer, MultiTaskUNet
from src.utils import append_jsonl, fmt_seconds, now_iso

logger = logging.getLogger(__name__)


class NonFiniteMetricsError(RuntimeError):
    """Raised when an epoch/batch metric is non-finite (NaN/Inf).

    Zero-silent-fallback mandate: a non-finite training or validation metric
    is FATAL, never a soft "restore best and end". The exception carries a
    structured diagnosis (first non-finite batch, raw batch loss, LR,
    AMP/scaler state, grad-norm, img/mask batch stats) so the failure is loud
    and self-explaining instead of a masked traceback.
    """

    def __init__(self, phase: str, epoch: int, diagnosis: dict,
                 per_sample_info: dict | None = None,
                 ended_early: str = "nan"):
        self.phase = phase
        self.epoch = epoch
        self.diagnosis = diagnosis
        # Partial per-slice data captured before the non-finite batch (so a
        # NaN-ended run can still dump "what evaluated" as a partial artifact).
        self.per_sample_info = per_sample_info
        # Explicit early-termination reason stamped into the summary
        # (e.g. "nan"). Never a silent soft-end.
        self.ended_early = ended_early
        super().__init__(self._format())

    def _format(self) -> str:
        d = self.diagnosis
        lines = [
            f"FATAL: non-finite {self.phase} metric at epoch {self.epoch} "
            f"(zero-silent-fallback: run aborted, no soft restore).",
            f"  first_nonfinite_batch : {d.get('first_nonfinite_batch')}",
            f"  raw_batch_loss        : {d.get('raw_batch_loss')}",
            f"  lr                    : {d.get('lr')}",
            f"  amp_dtype             : {d.get('amp_dtype')}",
            f"  amp_enabled           : {d.get('amp_enabled')}",
            f"  scaler_enabled        : {d.get('scaler_enabled')}",
            f"  scaler_scale          : {d.get('scaler_scale')}",
            f"  scaler_growth_factor  : {d.get('scaler_growth_factor')}",
            f"  scaler_growth_cnt     : {d.get('scaler_growth_cnt')}",
            f"  scaler_backoff_cnt    : {d.get('scaler_backoff_cnt')}",
            f"  grad_norm             : {d.get('grad_norm')}",
            f"  img_min/max/any_nan   : {d.get('img_min')} / {d.get('img_max')} / {d.get('img_any_nan')}",
            f"  mask_min/max/any_nan  : {d.get('mask_min')} / {d.get('mask_max')} / {d.get('mask_any_nan')}",
            f"  batch_size            : {d.get('batch_size')}",
            f"  compile_active        : {d.get('compile_active')}",
            f"  torch_version         : {d.get('torch_version')}",
            f"  cuda_version          : {d.get('cuda_version')}",
        ]
        return "\n".join(lines)


def _build_nan_diagnosis(
    phase: str,
    epoch: int,
    batch_idx: int | None,
    raw_batch_loss,
    lr,
    scaler,
    grad_norm,
    images,
    masks,
    batch_size: int,
    compile_active: bool,
    output_probe=None,
) -> dict:
    """Build a structured, JSON-serializable diagnosis of a non-finite metric.

    Captures exactly what is needed to discriminate the root-cause hypotheses
    (AMP fp16/bf16 overflow, torch.compile miscompile, LR/scaler, bad data
    batch): the first non-finite batch, its raw loss, the current LR, the
    AMP/scaler state (dtype, scale, growth/backoff counters), the grad-norm if
    computed, and per-batch img/mask statistics (min/max/any-NaN).
    """
    def _scalar(x):
        if x is None:
            return None
        try:
            # Detach tensors before float() to avoid the "Converting a tensor
            # to a Python scalar" UserWarning (and any grad-graph side effects)
            # when the value is a 0-dim / 1-elem tensor requiring grad.
            if torch.is_tensor(x):
                return float(x.detach())
            return float(x)
        except (TypeError, ValueError):
            return None

    def _tensor_stats(t):
        if t is None:
            return None, None, None
        try:
            with torch.no_grad():
                return (
                    float(t.min()),
                    float(t.max()),
                    bool(torch.isnan(t).any()),
                )
        except Exception:
            return None, None, None

    img_min, img_max, img_nan = _tensor_stats(images)
    mask_min, mask_max, mask_nan = _tensor_stats(masks)

    amp_dtype = "bfloat16"  # matches torch.autocast(dtype=torch.bfloat16) in _run_epoch
    amp_enabled = (scaler is not None) and bool(getattr(scaler, "_enabled", False))
    scaler_scale = None
    scaler_growth_factor = None
    scaler_growth_cnt = None
    scaler_backoff_cnt = None
    if scaler is not None:
        try:
            scaler_scale = float(scaler._scale.item())
        except Exception:
            scaler_scale = None
        scaler_growth_factor = getattr(scaler, "_growth_factor", None)
        scaler_growth_cnt = getattr(scaler, "_growth_cnt", None)
        scaler_backoff_cnt = getattr(scaler, "_backoff_cnt", None)

    return {
        "phase": phase,
        "epoch": int(epoch),
        "first_nonfinite_batch": int(batch_idx) if batch_idx is not None else None,
        "raw_batch_loss": _scalar(raw_batch_loss),
        "lr": _scalar(lr),
        "amp_dtype": amp_dtype,
        "amp_enabled": bool(amp_enabled),
        "scaler_enabled": bool(amp_enabled),
        "scaler_scale": scaler_scale,
        "scaler_growth_factor": scaler_growth_factor,
        "scaler_growth_cnt": scaler_growth_cnt,
        "scaler_backoff_cnt": scaler_backoff_cnt,
        "grad_norm": _scalar(grad_norm),
        "img_min": img_min,
        "img_max": img_max,
        "img_any_nan": img_nan,
        "mask_min": mask_min,
        "mask_max": mask_max,
        "mask_any_nan": mask_nan,
        "batch_size": int(batch_size),
        "compile_active": bool(compile_active),
        # Non-finite element counts per model output head at the fatal batch
        # (lazy probe, computed only on the raise path). The loss criteria are
        # numerically stable, so a NaN/Inf loss with clean inputs can only
        # originate from non-finite forward outputs; this attributes the
        # divergence to seg_out vs cls_out.
        "output_probe": output_probe,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _per_run_epoch_log_path(run_label: str) -> Path:
    """Per-run epoch log under results/round2/<run_label>/epoch_log.jsonl."""
    return RESULTS_DIR / "round2" / run_label / "epoch_log.jsonl"


def _dump_per_slice_dice(
    run_label: str,
    dataset: str,
    encoder: str,
    fold: int,
    seed: int,
    per_sample_info: dict,
) -> Path:
    """Write per-slice Dice records to results/round2/<run_label>/per_slice_dice.jsonl.

    Each record: {run_label, dataset, encoder, fold, seed, case_id, dice,
    empty_pred, empty_gt, label_int}. ``fold`` is -1 when no fold is active.
    """
    out_path = RESULTS_DIR / "round2" / run_label / "per_slice_dice.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    case_ids = per_sample_info.get("case_ids") or []
    dice = per_sample_info.get("dice") or []
    empty_pred = per_sample_info.get("empty_pred") or []
    empty_gt = per_sample_info.get("empty_gt") or []
    labels = per_sample_info.get("labels") or []
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(len(dice)):
            record = {
                "run_label": run_label,
                "dataset": dataset,
                "encoder": encoder,
                "fold": int(fold),
                "seed": int(seed),
                "case_id": case_ids[i] if i < len(case_ids) else str(i),
                "dice": float(dice[i]),
                "empty_pred": bool(empty_pred[i]) if i < len(empty_pred) else False,
                "empty_gt": bool(empty_gt[i]) if i < len(empty_gt) else False,
                "label_int": int(labels[i]) if i < len(labels) else -1,
            }
            f.write(json.dumps(record) + "\n")
    return out_path


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
    case_ids: list | None = None,
    collect_per_sample: bool = False,
    lr: float | None = None,
    compile_active: bool = False,
    epoch: int = 1,
    run_label: str | None = None,
    # Fold-level non-finite-grad skip tally, owned by the caller
    # (train_single_run) as a one-element list so the GRAD_SKIP_MAX_PER_FOLD
    # cap and the summary stamp count across ALL epochs of the fold, not
    # per-epoch. Caller may omit it (None) — the cap then degrades to the
    # per-epoch local count.
    skip_counter: list | None = None,
    # Canonical GradNorm (Chen et al. 2018) -- only used when
    # ``gradnorm_mode == "canonical"``. ``canonical_weights`` is a raw
    # nn.Parameter([w_seg, w_cls]) updated by a DEDICATED optimizer
    # (``canonical_optimizer``); the model never receives L_grad gradients.
    # ``canonical_alpha`` is the GradNorm asymmetry exponent.
    # ``canonical_state`` is a mutable dict (owned by the caller) holding
    # ``initial_losses``; it is set once on the first batch and persists
    # across epochs, mirroring the reference probe.
    canonical_weights: nn.Parameter | None = None,
    canonical_optimizer: optim.Optimizer | None = None,
    canonical_alpha: float = 1.5,
    canonical_state: dict | None = None,
):
    """Run one training or validation epoch.

    Args:
        case_ids: Optional list of case identifiers aligned with the dataset
            order (used only when ``collect_per_sample`` is True and the
            loader is not shuffled, so batch order maps 1:1 to dataset order).
        collect_per_sample: When True (validation), collect per-sample Dice
            scores and case ids for downstream per-slice result dumps.
        lr: Current learning rate (for non-finite diagnosis).
        compile_active: Whether torch.compile is active (for diagnosis).
        epoch: 1-based epoch number (for non-finite diagnosis).

    Returns:
        (mean_loss, accuracy, mean_dice, per_sample_info) where
        ``per_sample_info`` is None in train mode, or a dict with keys
        ``case_ids``, ``dice``, ``empty_pred``, ``empty_gt``, ``labels``
        (each a list aligned per sample) in val mode.

    Raises:
        NonFiniteMetricsError: if any batch loss or the epoch mean metric is
            non-finite (zero-silent-fallback: FATAL, never a soft skip/end).
    """
    model.train() if train else model.eval()

    total_loss = 0.0
    total = 0
    correct = 0
    dice_vals: list[float] = []
    steps = 0
    clip_engaged = 0
    clip_max_norm = 0.0
    clip_sum_norm = 0.0
    nonfinite_grad_skips = 0
    last_grad_norm: float | None = None
    per_sample_dice: list[float] = []
    per_sample_empty_pred: list[bool] = []
    per_sample_empty_gt: list[bool] = []
    per_sample_labels: list[int] = []
    sample_offset = 0
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
                elif canonical_weights is not None:
                    # Canonical GradNorm: the combined loss uses DETACHED raw
                    # task weights so the model receives NO L_grad gradient
                    # (only the dedicated optimizer_weights updates the weights).
                    w = canonical_weights.detach()
                    loss = w[0] * seg_loss + w[1] * cls_loss
                else:
                    loss = seg_loss + cls_loss

            # Zero-silent-fallback: a non-finite batch loss is FATAL (never a
            # soft skip). Capture the FIRST non-finite batch with full diagnosis
            # plus the partial per-sample data that evaluated before it.
            if not torch.isfinite(loss):
                output_probe = {
                    "seg_out_nonfinite": int((~torch.isfinite(seg_out)).sum().item()),
                    "cls_out_nonfinite": int((~torch.isfinite(cls_out)).sum().item()),
                }
                partial_per_sample = None
                if collect_per_sample and not train and per_sample_dice:
                    partial_per_sample = {
                        "case_ids": (list(case_ids)[:len(per_sample_dice)]
                                     if case_ids is not None
                                     else [str(i) for i in range(len(per_sample_dice))]),
                        "dice": list(per_sample_dice),
                        "empty_pred": list(per_sample_empty_pred),
                        "empty_gt": list(per_sample_empty_gt),
                        "labels": list(per_sample_labels),
                    }
                raise NonFiniteMetricsError(
                    phase="train" if train else "val",
                    epoch=epoch,
                    diagnosis=_build_nan_diagnosis(
                        phase="train" if train else "val",
                        epoch=epoch,
                        batch_idx=batch_idx,
                        raw_batch_loss=loss,
                        lr=lr,
                        scaler=scaler,
                        grad_norm=last_grad_norm,
                        images=images,
                        masks=masks,
                        batch_size=int(images.size(0)),
                        compile_active=compile_active,
                        output_probe=output_probe,
                    ),
                    per_sample_info=partial_per_sample,
                )

            if train:

                if gradnorm is not None and not static_weights and not bool(gradnorm.has_initial_losses.item()):
                    gradnorm.set_initial_losses(seg_loss.detach(), cls_loss.detach())

                # Canonical GradNorm: record the first-batch per-task losses
                # (used for the relative-rate / target-norm computation).
                if (
                    canonical_weights is not None
                    and not static_weights
                    and canonical_state is not None
                    and canonical_state["initial_losses"] is None
                ):
                    canonical_state["initial_losses"] = torch.stack(
                        [seg_loss.detach(), cls_loss.detach()]
                    )

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

                # Canonical GradNorm (Chen et al. 2018): compute L_grad from the
                # SAME unscaled per-task norms (autograd.grad on the unscaled
                # losses, AMP-safe) and step the DEDICATED optimizer_weights.
                # The model optimizer is untouched here, so the model receives
                # no L_grad gradient (strict detachment).
                if canonical_weights is not None and not static_weights:
                    seg_grads = torch.autograd.grad(
                        seg_loss, shared_params, retain_graph=True, allow_unused=True
                    )
                    cls_grads = torch.autograd.grad(
                        cls_loss, shared_params, retain_graph=True, allow_unused=True
                    )

                    def _canonical_grad_norm(grads):
                        values = [g.norm() for g in grads if g is not None]
                        if not values:
                            return torch.tensor(0.0, device=device)
                        return torch.norm(torch.stack(values))

                    seg_norm = _canonical_grad_norm(list(seg_grads))
                    cls_norm = _canonical_grad_norm(list(cls_grads))
                    norms = torch.stack([seg_norm, cls_norm]).detach()

                    with torch.no_grad():
                        losses = torch.stack([seg_loss.detach(), cls_loss.detach()])
                        inv_rates = losses / (canonical_state["initial_losses"] + 1e-8)
                        inv_rates = inv_rates / inv_rates.mean().clamp(min=1e-8)
                        target = norms.mean() * (inv_rates ** canonical_alpha)

                    # NOTE: grad_loss is BUILT here (it references the LIVE
                    # canonical_weights) but its backward() + the dedicated
                    # optimizer_weights.step() + clamp/renorm are DEFERRED to
                    # after the model's backward+step (see below). The combined
                    # loss above uses w = canonical_weights.detach(), which
                    # shares the SAME storage/version-counter as
                    # canonical_weights; stepping the weights here (in-place)
                    # would bump that version counter before the model
                    # backward consumes the graph -> "modified by an inplace
                    # operation" RuntimeError. Deferring matches the reference
                    # ordering (scripts/run_canonical_gradnorm_panda.py
                    # L202-224): model backward+step FIRST, then weights.
                    grad_loss = torch.sum(
                        torch.abs(canonical_weights * norms - target)
                    )
                    canonical_optimizer.zero_grad(set_to_none=True)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(retain_graph=gradnorm is not None and not static_weights)
                if gradnorm_loss is not None:
                    scaler.scale(gradnorm_loss).backward()
                clip_params = list(model.parameters())
                if gradnorm is not None and not static_weights:
                    clip_params += list(gradnorm.parameters())
                # PROTOCOL: global grad-norm clip (documented in config + run
                # summary + paper §4.1). ``clip_grad_norm_`` returns the
                # PRE-clip total norm, so a value > GRAD_CLIP_MAX_NORM means the
                # clip actually engaged this step. Log it loudly (throttled) so
                # an exploding batch is visible in the run log, not just in the
                # post-hoc NaN diagnosis. Clipping bounds the per-step update so
                # a single 1e5+ norm cannot push the shared encoder into a
                # near-overflow regime; it does NOT suppress a genuine NaN (a
                # non-finite forward loss is still FATAL above).
                #
                # AMP ordering: scale -> backward -> UNSCALE -> clip -> step.
                # ``clip_grad_norm_`` must run on the TRUE (unscaled) grads;
                # clipping the 65536x-scaled grads would make the effective
                # clip max_norm/65536 per step. Guard for the no-AMP path
                # (scaler is None / disabled) where grads are already unscaled.
                if scaler is not None:
                    scaler.unscale_(optimizer)
                last_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        clip_params, max_norm=GRAD_CLIP_MAX_NORM
                    ).item()
                )
                # Zero-silent-fallback, bounded-skip variant: occasional
                # non-finite GRADIENTS are expected under mixed precision —
                # Micikevicius et al. 2018 skip the update on overflow, and the
                # PyTorch GradScaler default skips the step when inf/NaN grads
                # are found. Such a batch is skipped LOUDLY (counted, logged,
                # excluded from epoch metrics, stamped into the summary) and is
                # FATAL only past GRAD_SKIP_MAX_PER_FOLD — a genuine divergence
                # blows the cap within a few batches and aborts exactly as the
                # old unconditional gate did. Non-finite LOSSES remain
                # unconditionally FATAL (check above).
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in clip_params
                ):
                    nonfinite_grad_skips += 1
                    if skip_counter is not None:
                        skip_counter[0] += 1
                        fold_skips = skip_counter[0]
                    else:  # defensive: callers without a counter
                        fold_skips = nonfinite_grad_skips
                    if fold_skips > GRAD_SKIP_MAX_PER_FOLD:
                        raise NonFiniteMetricsError(
                            phase="train",
                            epoch=epoch,
                            diagnosis=_build_nan_diagnosis(
                                phase="train",
                                epoch=epoch,
                                batch_idx=batch_idx,
                                raw_batch_loss=loss,
                                lr=lr,
                                scaler=scaler,
                                grad_norm=last_grad_norm,
                                images=images,
                                masks=masks,
                                batch_size=int(images.size(0)),
                                compile_active=compile_active,
                            ),
                        )
                    logger.warning(
                        "[%s] epoch %d batch %d: non-finite gradient(s) after "
                        "clip -> batch skipped, no update applied (skip "
                        "%d/%d; a non-finite loss would still be FATAL)",
                        run_label, epoch, batch_idx,
                        fold_skips, GRAD_SKIP_MAX_PER_FOLD,
                    )
                    # Drop the poisoned grads so nothing leaks into the next
                    # accumulation, then hand the overflow to the GradScaler's
                    # own path: scaler.update() reads the inf found_inf that
                    # unscale_ recorded, shrinks the loss scale (dynamic loss
                    # scaling), and RESETS the per-optimizer unscale/step state
                    # machine. Without it the state stays UNSCALED and the next
                    # batch's scaler.unscale_() raises "already been called
                    # since the last update()" (killed run 11 attempt 1).
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
                # Throttled clip telemetry (the PROTOCOL note above promises a
                # throttle): accumulate instead of a per-batch firehose; the
                # per-epoch aggregate is emitted after the batch loop and only
                # a far-above-threshold max stays loud.
                if last_grad_norm > GRAD_CLIP_MAX_NORM:
                    clip_engaged += 1
                clip_max_norm = max(clip_max_norm, last_grad_norm)
                clip_sum_norm += last_grad_norm
                scaler.step(optimizer)
                scaler.update()
                if gradnorm is not None and not static_weights:
                    gradnorm.normalize_()

                # Canonical GradNorm (Chen et al. 2018): DEDICATED
                # optimizer_weights step, DEFERRED to after the model's
                # backward+step (see the deferred grad_loss construction
                # above). The model backward has now CONSUMED the combined-loss
                # graph, so the in-place weight update (Adam step + clamp +
                # sum-2 renorm) can no longer bump the shared version counter
                # under a pending graph. The model never receives L_grad
                # (strict detachment); only optimizer_weights updates the
                # weights.
                if canonical_weights is not None and not static_weights:
                    grad_loss.backward()
                    canonical_optimizer.step()
                    # Renormalize weights so sum(w) = 2.0 (GRADNORM_WEIGHT_CLAMP
                    # bounds each weight to [1e-4, clamp], mirroring the
                    # parameterized path's clamp semantics).
                    with torch.no_grad():
                        canonical_weights.data = canonical_weights.data.clamp(
                            min=1e-4, max=GRADNORM_WEIGHT_CLAMP
                        )
                        canonical_weights.data = (
                            canonical_weights.data * (2.0 / canonical_weights.data.sum().clamp(min=1e-4))
                        )

            total_loss += float(loss.item()) if torch.isfinite(loss) else 0.0
            preds = cls_out.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
            dice_vals.append(dice_coefficient(seg_out.detach(), masks.detach(), seg_classes))
            steps += 1

            if collect_per_sample and not train:
                per = dice_coefficient_per_sample(seg_out.detach(), masks.detach(), seg_classes)
                per_sample_dice.extend(float(x) for x in per.tolist())
                if seg_classes == 1:
                    pred_bin = (torch.sigmoid(seg_out.detach()) > 0.5).float()
                    pred_sums = pred_bin.sum(dim=(1, 2, 3))
                    gt_sums = masks.detach().sum(dim=(1, 2, 3))
                else:
                    pred_bin = torch.argmax(seg_out.detach(), dim=1)
                    pred_sums = (pred_bin > 0).float().sum(dim=(1, 2) if pred_bin.ndim == 3 else (1, 2, 3))
                    gt_sums = (masks.detach() > 0).float().sum(dim=(1, 2) if masks.detach().ndim == 3 else (1, 2, 3))
                per_sample_empty_pred.extend(bool(x) for x in (pred_sums == 0).tolist())
                per_sample_empty_gt.extend(bool(x) for x in (gt_sums == 0).tolist())
                per_sample_labels.extend(int(x) for x in labels.detach().cpu().tolist())
                sample_offset += int(labels.size(0))

    if train:
        logger.info(
            "[%s] epoch %d: clip engaged %d/%d batches; non-finite-grad "
            "skips %d; true grad-norm max=%.1f mean=%.1f (max_norm=%.3f)",
            run_label, epoch, clip_engaged, steps, nonfinite_grad_skips,
            clip_max_norm,
            (clip_sum_norm / steps) if steps else 0.0,
            GRAD_CLIP_MAX_NORM,
        )
        if clip_max_norm > 100.0:
            logger.warning(
                "[%s] epoch %d: max grad-norm %.1f far exceeds clip "
                "max_norm=%.3f (possible instability; per-step updates "
                "were still bounded)",
                run_label, epoch, clip_max_norm, GRAD_CLIP_MAX_NORM,
            )

    if steps == 0 or total == 0:
        # No batches were consumed (e.g. empty loader). This is a data/loader
        # defect, not a numeric one: fail loud with a clear message.
        raise NonFiniteMetricsError(
            phase="train" if train else "val",
            epoch=epoch,
            diagnosis={
                "phase": "train" if train else "val",
                "epoch": int(epoch),
                "first_nonfinite_batch": None,
                "raw_batch_loss": None,
                "lr": lr,
                "amp_dtype": "bfloat16",
                "amp_enabled": bool(getattr(scaler, "_enabled", False)) if scaler is not None else False,
                "scaler_enabled": bool(getattr(scaler, "_enabled", False)) if scaler is not None else False,
                "scaler_scale": None,
                "scaler_growth_factor": None,
                "scaler_growth_cnt": None,
                "scaler_backoff_cnt": None,
                "grad_norm": None,
                "img_min": None,
                "img_max": None,
                "img_any_nan": None,
                "mask_min": None,
                "mask_max": None,
                "mask_any_nan": None,
                "batch_size": 0,
                "compile_active": bool(compile_active),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "note": "no batches consumed (empty loader / 0 steps)",
            },
        )

    mean_loss = total_loss / steps
    mean_acc = correct / total
    mean_dice = float(np.mean(dice_vals))

    # Zero-silent-fallback: a non-finite EPOCH MEAN metric is FATAL. (A single
    # non-finite batch is already caught above; this guards the aggregate.)
    if not (math.isfinite(mean_loss) and math.isfinite(mean_acc) and math.isfinite(mean_dice)):
        raise NonFiniteMetricsError(
            phase="train" if train else "val",
            epoch=epoch,
            diagnosis={
                "phase": "train" if train else "val",
                "epoch": int(epoch),
                "first_nonfinite_batch": None,
                "raw_batch_loss": None,
                "lr": lr,
                "amp_dtype": "bfloat16",
                "amp_enabled": bool(getattr(scaler, "_enabled", False)) if scaler is not None else False,
                "scaler_enabled": bool(getattr(scaler, "_enabled", False)) if scaler is not None else False,
                "scaler_scale": float(scaler._scale.item()) if scaler is not None else None,
                "scaler_growth_factor": getattr(scaler, "_growth_factor", None) if scaler is not None else None,
                "scaler_growth_cnt": getattr(scaler, "_growth_cnt", None) if scaler is not None else None,
                "scaler_backoff_cnt": getattr(scaler, "_backoff_cnt", None) if scaler is not None else None,
                "grad_norm": last_grad_norm,
                "img_min": None,
                "img_max": None,
                "img_any_nan": None,
                "mask_min": None,
                "mask_max": None,
                "mask_any_nan": None,
                "batch_size": int(total / max(steps, 1)),
                "compile_active": bool(compile_active),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "note": "epoch-mean metric non-finite (aggregate); per-batch check did not fire",
                "mean_loss": mean_loss,
                "mean_acc": mean_acc,
                "mean_dice": mean_dice,
            },
        )

    per_sample_info = None
    if collect_per_sample and not train:
        per_sample_info = {
            "case_ids": list(case_ids) if case_ids is not None else [str(i) for i in range(len(per_sample_dice))],
            "dice": per_sample_dice,
            "empty_pred": per_sample_empty_pred,
            "empty_gt": per_sample_empty_gt,
            "labels": per_sample_labels,
        }

    return mean_loss, mean_acc, mean_dice, per_sample_info


def _deterministic_run_paths(run_label: str) -> tuple[Path, Path]:
    """Return deterministic (best-checkpoint, training-state) paths for a run.

    F-23: artifact names are derived purely from ``run_label`` (which already
    encodes the fold when k-fold CV is active, e.g. ``<label>_fold2of5``).
    No timestamps, no randomization: the same ``run_label`` always maps to the
    same two paths, so a resumed run can find and load prior artifacts.

        results/round2/<run_label>/best.pt
        results/round2/<run_label>/final.state.pt
    """
    run_dir = RESULTS_DIR / "round2" / run_label
    return run_dir / "best.pt", run_dir / "final.state.pt"


def _build_run_fingerprint(
    dataset: str,
    encoder: str,
    args,
    batch_size: int,
    fold_idx: int | None,
    k_folds: int | None,
    run_label: str,
) -> dict:
    """Config/dataset identity stamped into the state file (F-23).

    A resumed run rebuilds this from its own CLI args and compares it against
    the fingerprint stored in the state file. A mismatch means the state was
    minted by a *different* run (different dataset/encoder/seed/split) and
    must be rejected loudly rather than silently loaded.
    """
    return {
        "dataset": dataset,
        "encoder": encoder,
        "seed": int(getattr(args, "seed", -1)),
        "epochs": int(getattr(args, "epochs", -1)),
        "batch_size": int(batch_size),
        "k_folds": int(k_folds) if k_folds is not None else None,
        "fold_idx": int(fold_idx) if fold_idx is not None else None,
        "run_label": run_label,
        # GradNorm formulation (parameterized vs canonical). Stamped so a
        # canonical run and a parameterized run with the same label are
        # distinguished on resume (distinct deterministic run paths).
        "gradnorm_mode": getattr(args, "gradnorm_mode", "parameterized"),
    }


def _resolve_resume(
    model,
    optimizer,
    gradnorm,
    state_path: Path,
    device: str,
    fingerprint: dict,
    run_label: str,
    resume_enabled: bool,
) -> tuple[dict | None, str]:
    """Decide the resume branch for a run and load state when valid (F-23).

    Returns ``(state, branch)`` where ``branch`` is one of:
      - ``"resumed-from-epoch"``  -- a valid state file was found and loaded.
      - ``"fresh-start-no-state"``-- no state file present (or --no-resume).

    Fail-fast: if a state file is present but is corrupt/unloadable, or its
    fingerprint does not match the current run, this raises ``RuntimeError``
    (FATAL) instead of silently starting fresh.
    """
    # --no-resume: deliberate fresh start, never touch prior state.
    if not resume_enabled:
        logger.info(
            "[%s] RESUME BRANCH: fresh-start-no-state (--no-resume; prior state ignored)",
            run_label,
        )
        return None, "fresh-start-no-state"

    if not state_path.exists():
        logger.info(
            "[%s] RESUME BRANCH: fresh-start-no-state (no state file at %s)",
            run_label, state_path,
        )
        return None, "fresh-start-no-state"

    # State file present: it MUST load cleanly and match this run, else FATAL.
    try:
        state = load_training_state(model, optimizer, gradnorm, state_path, device)
    except Exception as ex:  # corrupt / truncated / wrong-torch-format state
        raise RuntimeError(
            f"FATAL: cannot load training state for resume at {state_path} "
            f"(corrupt or unreadable): {ex!r}. Refusing to silently start fresh."
        ) from ex

    if state is None:
        raise RuntimeError(
            f"FATAL: training state at {state_path} is present but has no "
            f"model_state; refusing to silently start fresh."
        )

    stored_fp = state.get("fingerprint")
    if stored_fp is not None:
        for key in ("dataset", "encoder", "seed", "k_folds", "fold_idx", "run_label"):
            if stored_fp.get(key) != fingerprint.get(key):
                raise RuntimeError(
                    f"FATAL: training state at {state_path} was minted by a "
                    f"different run (fingerprint mismatch on '{key}': "
                    f"stored={stored_fp.get(key)!r} vs current={fingerprint.get(key)!r}). "
                    f"Refusing to resume."
                )

    start_epoch = int(state.get("epoch", 0)) + 1
    logger.info(
        "[%s] RESUME BRANCH: resumed-from-epoch (loading %s, continuing at epoch %d)",
        run_label, state_path, start_epoch,
    )
    return state, "resumed-from-epoch"


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
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
    fold_idx: int | None = None,
    k_folds: int | None = None,
    splitter_branch: str | None = None,
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
        skip_connections: Ablation flag to zero UNet skip connections.
        static_weights: Use fixed lambda weights without dynamic balance.
        smoke_test: Run fast 1-epoch 2-batch smoke test.
        run_label: Run label identifier.
        train_idx: Optional precomputed training indices (for K-Fold CV).
        val_idx: Optional precomputed validation indices (for K-Fold CV).
        fold_idx: Optional fold index (0-indexed).
        k_folds: Optional total number of folds.

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

    if train_idx is None or val_idx is None:
        _split_result = make_group_split(
            labels, groups, seed=args.seed, test_size=0.2,
            grouping=meta.get("grouping"),
            provenance_path=DATASET_ROOTS[dataset] / "preprocessed" / "grouping_provenance.json",
        )
        train_idx, val_idx = _split_result
        if splitter_branch is None:
            splitter_branch = getattr(_split_result, "metadata", {}).get("branch")
    else:
        # Zero-leakage check on supplied fold indices
        tr_g = set(groups[train_idx])
        vl_g = set(groups[val_idx])
        overlap = tr_g & vl_g
        if overlap:
            raise RuntimeError(f"FATAL LEAKAGE DEFECT: fold {fold_idx} has {len(overlap)} overlapping groups!")

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
    if smoke_test:
        effective_workers = 0
        prefetch_factor = 0
        persistent_workers = False
        train_cache_size = 0
    else:
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
        gradnorm_mode = getattr(args, "gradnorm_mode", "parameterized")
        gradnorm = None
        canonical_weights = None
        canonical_optimizer = None
        canonical_state = None
        if use_gradnorm:
            if gradnorm_mode == "canonical":
                # Canonical GradNorm (Chen et al. 2018): raw task-weight
                # parameters in a DEDICATED optimizer (mirrors
                # scripts/run_canonical_gradnorm_panda.py L151-224). Initialized
                # at the static 5:1 ratio normalized to sum=2: [5/3, 1/3].
                # The model optimizer below does NOT include these weights, so
                # the model receives no L_grad gradient.
                canonical_weights = nn.Parameter(
                    torch.tensor([5.0 / 3.0, 1.0 / 3.0], dtype=torch.float32, device=device)
                )
                canonical_optimizer = optim.Adam([canonical_weights], lr=0.025)
                canonical_state = {"initial_losses": None}
            else:
                # Parameterized GradNorm (default): log-weights in the primary
                # Adam optimizer (unchanged behavior).
                gradnorm = GradNormBalancer(
                    args.lambda_seg, args.lambda_cls, getattr(args, "gradnorm_alpha", 1.5),
                    weight_clamp=GRADNORM_WEIGHT_CLAMP,
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

        # F-23: deterministic artifact paths derived purely from run_label
        # (which already encodes the fold for k-fold CV). No timestamps.
        ckpt_path, state_path = _deterministic_run_paths(run_label)
        fingerprint = _build_run_fingerprint(
            dataset, encoder, args, batch_size, fold_idx, k_folds, run_label,
        )
        start_epoch = 1
        last_completed_epoch = 0
        resume_state = None
        fold_skip_counter = [0]  # fold-level non-finite-grad skip tally (cap GRAD_SKIP_MAX_PER_FOLD)
        resume_branch = "fresh-start-no-state"

        logger.info(
            "[%d/%d] %s x %s | samples=%d train=%d val=%d bs=%d | "
            "workers=%d prefetch=%d persistent=%s cache=%d",
            run_index, total_runs, dataset.upper(), encoder,
            len(images), len(train_idx), len(val_idx), batch_size,
            effective_workers, prefetch_factor, persistent_workers, train_cache_size,
        )

        # F-23: single resume decision point. Loads the exact deterministic
        # state path when present; FATAL on corrupt/mismatch; loud branch log.
        resume_state, resume_branch = _resolve_resume(
            model, optimizer, gradnorm, state_path, device,
            fingerprint, run_label, resume_enabled=bool(getattr(args, "resume", False)),
        )

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
                tr_loss, tr_acc, tr_dice, _ = _run_epoch(
                    model, train_loader, optimizer, seg_criterion, cls_criterion,
                    device, scaler, gradnorm, meta["seg_classes"], train=True,
                    skip_counter=fold_skip_counter,
                    static_weights=static_weights,
                    lambda_seg=args.lambda_seg,
                    lambda_cls=args.lambda_cls,
                    smoke_test=smoke_test,
                    lr=args.lr,
                    compile_active=compile_active,
                    epoch=epoch,
                    run_label=run_label,
                    canonical_weights=canonical_weights,
                    canonical_optimizer=canonical_optimizer,
                    canonical_alpha=getattr(args, "gradnorm_alpha", 1.5),
                    canonical_state=canonical_state,
                )
                vl_loss, vl_acc, vl_dice, _ = _run_epoch(
                    model, val_loader, optimizer, seg_criterion, cls_criterion,
                    device, scaler, gradnorm, meta["seg_classes"], train=False,
                    static_weights=static_weights,
                    lambda_seg=args.lambda_seg,
                    lambda_cls=args.lambda_cls,
                    smoke_test=smoke_test,
                    lr=args.lr,
                    compile_active=compile_active,
                    epoch=epoch,
                    run_label=run_label,
                    canonical_weights=canonical_weights,
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

                epoch_record = {
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
                    # F-10 stamping: run identity + splitter provenance.
                    "run_label": run_label,
                    "fold": int(fold_idx) if fold_idx is not None else -1,
                    "seed": int(getattr(args, "seed", -1)),
                    "splitter_branch": splitter_branch,
                }
                # Legacy shared epoch log (paper/figures/loaders.py depends on it).
                append_jsonl(epoch_log_file, epoch_record)
                # Per-run epoch log under results/round2/<run_label>/ (F-10).
                if run_label is not None:
                    append_jsonl(_per_run_epoch_log_path(run_label), epoch_record)

                if not smoke_test:
                    save_training_state(
                        model, optimizer, gradnorm, state_path, epoch,
                        best_val_loss, best_val_acc, best_val_dice,
                        best_monitor_metric, patience_ctr, batch_size,
                        fingerprint=fingerprint,
                    )

                # This epoch completed cleanly (both train and val finite).
                last_completed_epoch = epoch

                if patience_ctr >= args.patience:
                    logger.info("Early stop at epoch %d (patience=%d)", epoch, args.patience)
                    break
                # NOTE: the old soft-end NaN check is GONE. A non-finite
                # train/val metric now raises NonFiniteMetricsError from
                # _run_epoch (zero-silent-fallback: FATAL, never a soft end).

            if smoke_test:
                final_acc, final_dice = vl_acc, vl_dice
            else:
                # Defect 2: the final-eval best.pt load is CONDITIONAL. A run
                # that ended early (e.g. NaN) may never have written best.pt,
                # so an unconditional load would crash with FileNotFoundError
                # and mask the real diagnosis. Only load when the file exists.
                if ckpt_path.exists():
                    load_checkpoint(model, ckpt_path, device)
                    _, final_acc, final_dice, final_per_sample = _run_epoch(
                        model, val_loader, optimizer, seg_criterion, cls_criterion,
                        device, scaler, gradnorm, meta["seg_classes"], train=False,
                        static_weights=static_weights,
                        lambda_seg=args.lambda_seg,
                        lambda_cls=args.lambda_cls,
                        case_ids=list(val_ds.case_ids) if val_ds is not None else None,
                        collect_per_sample=True,
                        lr=args.lr,
                        compile_active=compile_active,
                        epoch=epoch,
                        run_label=run_label,
                        canonical_weights=canonical_weights,
                    )
                    # (3) Final best-checkpoint per-slice Dice dump.
                    if final_per_sample is not None and run_label is not None:
                        _dump_per_slice_dice(
                            run_label=run_label,
                            dataset=dataset,
                            encoder=encoder,
                            fold=int(fold_idx) if fold_idx is not None else -1,
                            seed=int(getattr(args, "seed", -1)),
                            per_sample_info=final_per_sample,
                        )
                else:
                    # No best checkpoint was ever written (e.g. the run ended
                    # before any epoch produced a finite best). Fall back to the
                    # current (last) model weights for a best-effort final eval
                    # so the run still yields a summary + partial artifacts.
                    logger.warning(
                        "[%s] best.pt not found at %s (run ended before a best "
                        "checkpoint was written); using current weights for final eval.",
                        run_label, ckpt_path,
                    )
                    _, final_acc, final_dice, final_per_sample = _run_epoch(
                        model, val_loader, optimizer, seg_criterion, cls_criterion,
                        device, scaler, gradnorm, meta["seg_classes"], train=False,
                        static_weights=static_weights,
                        lambda_seg=args.lambda_seg,
                        lambda_cls=args.lambda_cls,
                        case_ids=list(val_ds.case_ids) if val_ds is not None else None,
                        collect_per_sample=True,
                        lr=args.lr,
                        compile_active=compile_active,
                        epoch=epoch,
                        run_label=run_label,
                        canonical_weights=canonical_weights,
                    )
                    if final_per_sample is not None and run_label is not None:
                        _dump_per_slice_dice(
                            run_label=run_label,
                            dataset=dataset,
                            encoder=encoder,
                            fold=int(fold_idx) if fold_idx is not None else -1,
                            seed=int(getattr(args, "seed", -1)),
                            per_sample_info=final_per_sample,
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
                "fold_idx": int(fold_idx) if fold_idx is not None else None,
                "k_folds": int(k_folds) if k_folds is not None else None,
                "checkpoint": str(ckpt_path),
                "state_path": str(state_path),
                "resume_branch": resume_branch,
                "resumed_from_epoch": int(start_epoch - 1) if resume_state is not None else 0,
                "duration_sec": float(duration),
                "duration_hms": fmt_seconds(duration),
                "attempt": attempt,
                "compile_enabled": bool(compile_active),
                "compile_backend": REPRO_TORCH_COMPILE_BACKEND or None,
                "compile_mode": "max-autotune",
                "skip_connections_ablated": bool(skip_connections),
                "smoke_test": bool(smoke_test),
                # PROTOCOL parameters (paper §4.1): the exact stabilizer settings
                # used by this run (global grad-norm clip + GradNorm weight clamp).
                "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
                "grad_skip_max_per_fold": GRAD_SKIP_MAX_PER_FOLD,
                "nonfinite_grad_skips": int(fold_skip_counter[0]),
                "gradnorm_weight_clamp": GRADNORM_WEIGHT_CLAMP,
                "gradnorm_mode": gradnorm_mode,
            }

        except NonFiniteMetricsError as ex:
            # Zero-silent-fallback: a non-finite metric is FATAL. Fail loud with
            # a structured diagnosis, write partial artifacts (per-slice dump of
            # what evaluated + an epoch-log record with ended_early: nan), and
            # return a result dict that main.py stamps into the summary with an
            # explicit ended_early: nan branch. No traceback masks the diagnosis.
            logger.error(
                "[%s] %s", run_label, ex,
            )
            diag = ex.diagnosis
            # (a) Partial per-slice dump of whatever evaluated before the
            #     non-finite batch (only when we have partial data).
            if ex.per_sample_info is not None and run_label is not None:
                try:
                    _dump_per_slice_dice(
                        run_label=run_label,
                        dataset=dataset,
                        encoder=encoder,
                        fold=int(fold_idx) if fold_idx is not None else -1,
                        seed=int(getattr(args, "seed", -1)),
                        per_sample_info=ex.per_sample_info,
                    )
                except Exception as dump_ex:
                    logger.warning("[%s] partial per-slice dump failed: %s", run_label, dump_ex)
            # (b) Epoch-log record with an explicit ended_early: nan branch.
            if run_label is not None:
                nan_record = {
                    "timestamp": now_iso(),
                    "dataset": dataset,
                    "encoder": encoder,
                    "epoch": int(ex.epoch),
                    "batch_size": int(batch_size),
                    "ended_early": "nan",
                    "nonfinite_phase": ex.phase,
                    "first_nonfinite_batch": diag.get("first_nonfinite_batch"),
                    "raw_batch_loss": diag.get("raw_batch_loss"),
                    "lr": diag.get("lr"),
                    "amp_dtype": diag.get("amp_dtype"),
                    "amp_enabled": diag.get("amp_enabled"),
                    "scaler_enabled": diag.get("scaler_enabled"),
                    "scaler_scale": diag.get("scaler_scale"),
                    "scaler_growth_factor": diag.get("scaler_growth_factor"),
                    "scaler_growth_cnt": diag.get("scaler_growth_cnt"),
                    "scaler_backoff_cnt": diag.get("scaler_backoff_cnt"),
                    "grad_norm": diag.get("grad_norm"),
                    "img_min": diag.get("img_min"),
                    "img_max": diag.get("img_max"),
                    "img_any_nan": diag.get("img_any_nan"),
                    "mask_min": diag.get("mask_min"),
                    "mask_max": diag.get("mask_max"),
                    "mask_any_nan": diag.get("mask_any_nan"),
                    "compile_active": diag.get("compile_active"),
                    "torch_version": diag.get("torch_version"),
                    "cuda_version": diag.get("cuda_version"),
                    "run_label": run_label,
                    "fold": int(fold_idx) if fold_idx is not None else -1,
                    "seed": int(getattr(args, "seed", -1)),
                    "splitter_branch": splitter_branch,
                    # PROTOCOL parameters (paper §4.1): the stabilizer settings
                    # active when this run hit the non-finite batch.
                    "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
                    "grad_skip_max_per_fold": GRAD_SKIP_MAX_PER_FOLD,
                    "nonfinite_grad_skips": int(fold_skip_counter[0]),
                    "gradnorm_weight_clamp": GRADNORM_WEIGHT_CLAMP,
                }
                append_jsonl(_per_run_epoch_log_path(run_label), nan_record)
                append_jsonl(epoch_log_file, nan_record)
            duration = time.time() - start_ts
            return {
                "status": "ended_early",
                "ended_early": "nan",
                "nonfinite_phase": ex.phase,
                "nonfinite_epoch": int(ex.epoch),
                "diagnosis": diag,
                "dataset": dataset,
                "encoder": encoder,
                "samples": int(len(images)),
                "train_samples": int(len(train_idx)),
                "val_samples": int(len(val_idx)),
                "batch_size": int(batch_size),
                "best_val_loss": float(best_val_loss),
                "best_val_acc": float(best_val_acc),
                "best_val_dice": float(best_val_dice),
                "final_val_acc": (float(vl_acc) if "vl_acc" in locals() and vl_acc is not None else None),
                "final_val_dice": (float(vl_dice) if "vl_dice" in locals() and vl_dice is not None else None),
                "last_completed_epoch": int(last_completed_epoch),
                "fold_idx": int(fold_idx) if fold_idx is not None else None,
                "k_folds": int(k_folds) if k_folds is not None else None,
                "checkpoint": str(ckpt_path),
                "state_path": str(state_path),
                "resume_branch": resume_branch,
                "resumed_from_epoch": int(start_epoch - 1) if resume_state is not None else 0,
                "duration_sec": float(duration),
                "duration_hms": fmt_seconds(duration),
                "attempt": attempt,
                "compile_enabled": bool(compile_active),
                "compile_backend": REPRO_TORCH_COMPILE_BACKEND or None,
                "compile_mode": "max-autotune",
                "skip_connections_ablated": bool(skip_connections),
                "smoke_test": bool(smoke_test),
                # PROTOCOL parameters (paper §4.1): the exact stabilizer settings
                # used by this run (global grad-norm clip + GradNorm weight clamp).
                "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
                "grad_skip_max_per_fold": GRAD_SKIP_MAX_PER_FOLD,
                "nonfinite_grad_skips": int(fold_skip_counter[0]),
                "gradnorm_weight_clamp": GRADNORM_WEIGHT_CLAMP,
                "gradnorm_mode": gradnorm_mode,
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
            for obj in (model, optimizer, scaler, seg_criterion, cls_criterion, train_loader, val_loader, train_ds, val_ds,
                        canonical_weights, canonical_optimizer):
                if obj is not None:
                    del obj
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as ex:
                    logger.warning("CUDA cache cleanup skipped: %s", ex)

    raise RuntimeError(f"Failed to train {dataset} x {encoder}: minimum batch size exhausted")


def train_kfold_cv(
    dataset: str,
    encoder: str,
    bundle: dict,
    meta: dict,
    args,
    device: str,
    epoch_log_file: Path,
    k_folds: int = 5,
    skip_connections: bool = False,
    static_weights: bool = False,
    smoke_test: bool = False,
    run_label: str | None = None,
) -> dict:
    """Run full K-fold cross-validation with zero patient leakage and compute mean +/- std metrics.

    Args:
        dataset: Dataset key (tcga, panda, siim, pannuke).
        encoder: Encoder backbone name.
        bundle: Pre-loaded dataset bundle from ``load_dataset_bundle``.
        meta: Dataset metadata from ``DATASET_META``.
        args: Parsed CLI arguments.
        device: 'cuda' or 'cpu'.
        epoch_log_file: Path for JSONL epoch logging.
        k_folds: Number of folds (default 5).
        skip_connections: Ablation flag for UNet skip connections.
        static_weights: Static loss balancing flag.
        smoke_test: Fast smoke test mode.
        run_label: Base run label identifier.

    Returns:
        Consolidated dict with per-fold results, mean, and standard deviation.
    """
    labels = bundle["labels"]
    groups = bundle["groups"]
    splits = make_group_kfold_splits(
        labels, groups, n_splits=k_folds, seed=args.seed,
        grouping=meta.get("grouping"),
        provenance_path=DATASET_ROOTS[dataset] / "preprocessed" / "grouping_provenance.json",
    )

    base_label = run_label or f"{dataset}_{encoder}"
    fold_results = []
    target_folds = [int(args.fold)] if getattr(args, "fold", None) is not None else list(range(k_folds))

    logger.info(
        "================================================================================"
    )
    logger.info(
        "STARTING %d-FOLD GROUP-AWARE CROSS-VALIDATION: %s x %s (target folds: %s)",
        k_folds, dataset.upper(), encoder, target_folds,
    )
    logger.info(
        "================================================================================"
    )

    for fold_idx in target_folds:
        tr_idx, vl_idx = splits[fold_idx]
        fold_run_label = f"{base_label}_fold{fold_idx + 1}of{k_folds}"
        logger.info(
            ">>> Running Fold %d/%d (train=%d samples, val=%d samples) <<<",
            fold_idx + 1, k_folds, len(tr_idx), len(vl_idx)
        )

        fold_res = train_single_run(
            dataset=dataset,
            encoder=encoder,
            bundle=bundle,
            meta=meta,
            args=args,
            device=device,
            epoch_log_file=epoch_log_file,
            run_index=fold_idx + 1,
            total_runs=len(target_folds),
            skip_connections=skip_connections,
            static_weights=static_weights,
            smoke_test=smoke_test,
            run_label=fold_run_label,
            train_idx=tr_idx,
            val_idx=vl_idx,
            fold_idx=fold_idx,
            k_folds=k_folds,
            splitter_branch=getattr(splits, "metadata", {}).get("branch"),
        )
        fold_results.append(fold_res)

    val_accs = [r["final_val_acc"] for r in fold_results]
    val_dices = [r["final_val_dice"] for r in fold_results]
    val_losses = [r["best_val_loss"] for r in fold_results]

    mean_acc = float(np.mean(val_accs))
    std_acc = float(np.std(val_accs, ddof=1)) if len(val_accs) > 1 else 0.0
    mean_dice = float(np.mean(val_dices))
    std_dice = float(np.std(val_dices, ddof=1)) if len(val_dices) > 1 else 0.0
    mean_loss = float(np.mean(val_losses))
    std_loss = float(np.std(val_losses, ddof=1)) if len(val_losses) > 1 else 0.0

    logger.info("================================================================================")
    logger.info("CROSS-VALIDATION SUMMARY [%s x %s (%d folds completed)]:", dataset.upper(), encoder, len(fold_results))
    for idx, r in enumerate(fold_results):
        logger.info("  Fold %d: Acc = %.4f, Dice = %.4f, ValLoss = %.4f", idx + 1, r["final_val_acc"], r["final_val_dice"], r["best_val_loss"])
    logger.info("--------------------------------------------------------------------------------")
    logger.info("MEAN +/- STD: Acc = %.4f +/- %.4f | Dice = %.4f +/- %.4f | ValLoss = %.4f +/- %.4f",
                mean_acc, std_acc, mean_dice, std_dice, mean_loss, std_loss)
    logger.info("================================================================================")

    return {
        "status": "completed",
        "dataset": dataset,
        "encoder": encoder,
        "k_folds": k_folds,
        "completed_folds": len(fold_results),
        "mean_val_acc": mean_acc,
        "std_val_acc": std_acc,
        "mean_val_dice": mean_dice,
        "std_val_dice": std_dice,
        "mean_val_loss": mean_loss,
        "std_val_loss": std_loss,
        "fold_results": fold_results,
    }


def _compute_class_weights(labels: np.ndarray, num_classes: int, device: str) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    if present.any():
        weights[present] = counts.sum() / (num_classes * counts[present])
    return torch.tensor(weights, dtype=torch.float32, device=device)