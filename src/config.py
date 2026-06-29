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
DATASET_META = {
    "tcga": {
        "num_classes": 2,
        "img_size": 256,
        "seg_classes": 1,
        "binary_positive_min": 1,
        "use_macenko": False,
    },
    "panda": {
        "num_classes": 6,
        "img_size": 128,
        "seg_classes": 6,
        "use_macenko": True,
    },
    "siim": {
        "num_classes": 2,
        "img_size": 224,
        "seg_classes": 1,
        "binary_positive_min": 1,
        "use_macenko": False,
    },
    "pannuke": {
        "num_classes": 19,
        "img_size": 256,
        "seg_classes": 6,
        "use_macenko": True,
    },
}

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 32
RANDOM_SEED = 42
DEFAULT_ENCODERS = ["vgg16", "mobilenet_v2"]
DEFAULT_DATASETS = ["tcga", "panda", "siim", "pannuke"]

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