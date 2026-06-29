"""Checkpoint save/load logic.

Provides:
    - save_checkpoint: Save model state_dict.
    - load_checkpoint: Load model state_dict.
    - save_training_state: Save full training state for resuming.
    - load_training_state: Resume training state from checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(model, path: Path) -> None:
    """Save model state_dict to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save(state, path)


def load_checkpoint(model, path: Path, device: str) -> None:
    """Load model state_dict from *path*."""
    state = torch.load(path, map_location=device)
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state)
    else:
        model.load_state_dict(state)


def save_training_state(
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


def load_training_state(model, optimizer, gradnorm, path: Path, device: str) -> dict | None:
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