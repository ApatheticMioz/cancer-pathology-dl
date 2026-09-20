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
    model, optimizer, gradnorm, state_path: Path, epoch: int,
    best_val_loss: float, best_val_acc: float, best_val_dice: float,
    best_monitor_metric: float, patience_ctr: int, batch_size: int,
    fingerprint: dict | None = None,
) -> None:
    """Save full training state for resuming.

    ``state_path`` is the explicit, deterministic state-file path (F-23:
    ``results/round2/<run_label>/final.state.pt``) — no longer derived from
    the checkpoint name, so the state file is always found on resume.

    ``fingerprint`` is a config/dataset identity dict (dataset, encoder,
    seed, epochs, batch_size, k_folds, fold_idx, run_label) stamped into the
    state file so a resumed run can verify it is loading the state of the
    *same* run before trusting it (F-23).
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
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
            "fingerprint": fingerprint,
        },
        state_path,
    )


def load_training_state(model, optimizer, gradnorm, state_path: Path, device: str) -> dict | None:
    """Resume training state from an explicit state-file path. Returns state dict or None."""
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