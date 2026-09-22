"""Centralized configuration and hyperparameters for all experimental phases.

Defines dataset metadata, phase-specific hyperparameters (V1, V2, V2.1),
environment variable defaults, and file paths for checkpoints, logs, and
audit records.

Phase configurations per the research paper:
    V1  - Fixed loss weights (lambda_seg=5, lambda_cls=1), lr=1e-3,
          strict group-aware splits, no GradNorm.
    V2  - GradNorm dynamic loss balancing, Macenko normalization enabled,
          lr=1e-4, gradnorm_alpha=1.5.
    V2.1 - PanNuke control run using the V1 pipeline configuration.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
DATASET_AUDIT_FILE = CHECKPOINT_DIR / "dataset_audit.json"
EPOCH_LOG_FILE = CHECKPOINT_DIR / "epoch_log.jsonl"
RESULTS_DIR = BASE_DIR / "results"
REPRO_SUMMARY_FILE = CHECKPOINT_DIR / "optimized_summary.json"

# ---------------------------------------------------------------------------
# Environment variable defaults (centralized from main.py / training.py)
# ---------------------------------------------------------------------------
def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


REPRO_DISABLE_CUDNN = _env_bool("REPRO_DISABLE_CUDNN", default=False)
REPRO_STRICT_BATCH_CHECKS = _env_bool("REPRO_STRICT_BATCH_CHECKS", default=False)
REPRO_ALLOW_BIG_CACHE = _env_bool("REPRO_ALLOW_BIG_CACHE", default=False)
REPRO_ALLOW_UNC_WORKERS = _env_bool("REPRO_ALLOW_UNC_WORKERS", default=False)
REPRO_TORCH_COMPILE_BACKEND = os.getenv("REPRO_TORCH_COMPILE_BACKEND", "").strip() or None

# ---------------------------------------------------------------------------
# Dataset roots (relative to project root)
# ---------------------------------------------------------------------------
DATASET_ROOTS = {
    "tcga": BASE_DIR / "data/TCGA",
    "panda": BASE_DIR / "data/PANDA",
    "siim": BASE_DIR / "data/SIIM",
    "pannuke": BASE_DIR / "data/PanNuke",
}

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------
# ``grouping`` declares the semantic unit of the ``groups`` column for each
# dataset. The splitters (src/data.py) validate this declaration against the
# actual group cardinality:
#   - 'patient': groups are patient-level; the splitter REQUIRES
#     n_unique_groups < n_samples (multiple images per patient), else FATAL.
#   - 'image' / 'patch': groups are per-image / per-patch; trivial groups
#     (one sample per group) are expected and allowed.
#
# Release caveats:
#   - PANDA and SIIM releases have exactly ONE image per group unit, so their
#     group_id columns are trivial (one per sample). PANDA is declared 'image'
#     (consistent). SIIM is declared 'patient', but its on-disk index.csv
#     currently carries per-image group_ids, so the splitter FATALs until the
#     index is regenerated at patient level (scripts/prep_siim_full.py
#     --index-only).
#   - The PanNuke mirror has no slide IDs; groups are per-patch, so 'patch'
#     is the correct declaration and trivial groups are expected.
DATASET_META = {
    "tcga": {
        "num_classes": 2,
        "img_size": 256,
        "seg_classes": 1,
        "binary_positive_min": 1,
        "use_macenko": False,
        "grouping": "patient",
    },
    "panda": {
        "num_classes": 6,
        "img_size": 128,
        "seg_classes": 6,
        "use_macenko": True,
        "grouping": "image",
    },
    "siim": {
        "num_classes": 2,
        "img_size": 224,
        "seg_classes": 1,
        "binary_positive_min": 1,
        "use_macenko": False,
        "grouping": "patient",
    },
    "pannuke": {
        "num_classes": 19,
        "img_size": 256,
        "seg_classes": 6,
        "use_macenko": True,
        "grouping": "patch",
    },
}

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 32
RANDOM_SEED = 42
DEFAULT_ENCODERS = ["vgg16", "mobilenet_v2"]
DEFAULT_DATASETS = ["tcga", "panda", "siim", "pannuke"]

# Global gradient-norm clipping applied to every optimizer step (model + any
# GradNorm balancer parameters). This is a PROTOCOL parameter: it bounds the
# per-step update magnitude so a single exploding batch (e.g. a 1e5+ total
# grad norm on a hard TCGA batch) cannot push the shared encoder into a
# near-overflow regime. The value is stamped into every run summary and is
# referenced by the paper's §4.1 protocol sentence. It is a stabilizer, NOT a
# NaN suppressor: a non-finite metric is still FATAL (zero-silent-fallback).
GRAD_CLIP_MAX_NORM = 1.0
# Loud bounded skip for non-finite gradients after the clip: each skipped
# batch is counted, logged, excluded from epoch metrics, and stamped into the
# fold summary; breaching this cap is FATAL (same abort as the old unconditional
# gate). Backing: Micikevicius et al. 2018 (mixed-precision dynamic loss scaling
# skips the update on overflow; NVIDIA guidance: infrequent skips leave
# convergence unaffected) and the PyTorch GradScaler default (step skipped when
# inf/NaN grads are found). Non-finite LOSSES remain unconditionally FATAL.
GRAD_SKIP_MAX_PER_FOLD = 25

# GradNormBalancer log-weight clamp (applied in normalize_ before exp()).
# Bounds the task-weight imbalance so no single task's gradient can dominate
# the shared encoder into a near-overflow regime, and prevents exp() from
# overflowing/underflowing float32 (exp(>~88) -> inf, exp(<~-88) -> 0).
# Also a PROTOCOL parameter stamped into the run summary.
GRADNORM_WEIGHT_CLAMP = 10.0

REQUIRED_MATRIX = [
    ("tcga", "vgg16"),
    ("tcga", "mobilenet_v2"),
    ("panda", "vgg16"),
    ("panda", "mobilenet_v2"),
    ("siim", "vgg16"),
    ("siim", "mobilenet_v2"),
    ("pannuke", "vgg16"),
    ("pannuke", "mobilenet_v2"),
]

# ---------------------------------------------------------------------------
# Phase configurations
# ---------------------------------------------------------------------------
PHASE_CONFIGS = {
    "v1": {
        # Baseline: fixed loss weights, higher LR, no GradNorm
        "lr": 1e-3,
        "lambda_seg": 5.0,
        "lambda_cls": 1.0,
        "use_gradnorm": False,
        "gradnorm_alpha": 0.0,
        "epochs": 50,
        "patience": 10,
    },
    "v2": {
        # Enhanced: GradNorm dynamic balancing, Macenko normalization, lower LR
        "lr": 1e-4,
        "lambda_seg": 5.0,
        "lambda_cls": 1.0,
        "use_gradnorm": True,
        "gradnorm_alpha": 1.5,
        "epochs": 50,
        "patience": 10,
    },
    "v2.1": {
        # PanNuke control using V1 pipeline
        "lr": 1e-3,
        "lambda_seg": 5.0,
        "lambda_cls": 1.0,
        "use_gradnorm": False,
        "gradnorm_alpha": 0.0,
        "epochs": 50,
        "patience": 10,
    },
}

# ---------------------------------------------------------------------------
# Paper reference targets for comparison
# ---------------------------------------------------------------------------
PAPER_TARGETS = {
    "tcga_vgg16": {"acc": 0.89, "dice": 0.97},
    "tcga_mobilenet_v2": {"acc": 0.90, "dice": 0.98},
    "panda_vgg16": {"acc": 0.87, "dice": 0.98},
    "siim_vgg16": {"acc": 0.82, "dice": 0.99},
    "panda_mobilenet_v2": {"acc": 0.88, "dice": 0.99},
    "siim_mobilenet_v2": {"acc": 0.87, "dice": 0.99},
}

# ---------------------------------------------------------------------------
# External dataset competition references
# ---------------------------------------------------------------------------
PANDA_COMPETITION = "prostate-cancer-grade-assessment"
SIIM_COMPETITION = "siim-acr-pneumothorax-segmentation"