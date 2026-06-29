"""Centralized configuration and hyperparameters for all experimental phases.

Defines dataset metadata, phase-specific hyperparameters (V1, V2, V2.1),
and file paths for checkpoints, logs, and audit records.

Phase configurations per the research paper:
    V1  - Fixed loss weights (lambda_seg=5, lambda_cls=1), lr=1e-3,
          strict group-aware splits, no GradNorm.
    V2  - GradNorm dynamic loss balancing, Macenko normalization enabled,
          lr=1e-4, gradnorm_alpha=1.5.
    V2.1 - PanNuke control run using the V1 pipeline configuration.
"""
from __future__ import annotations

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
# External dataset download references
# ---------------------------------------------------------------------------
ISIC_URLS = {
    "ISIC2018_Task1-2_Training_Input.zip": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip",
    "ISIC2018_Task1_Training_GroundTruth.zip": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip",
    "ISIC2018_Task3_Training_GroundTruth.zip": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip",
}

ISIC_KAGGLE_FALLBACK_REFS = [
    "shonenkov/isic2018",
    "ceenen/isic-2018-challenge-task-1-segmentation",
    "tschandl/isic2018-challenge-task1-data-segmentation",
    "xxc025/isic2018",
    "yupanliu999/isic2018",
]

PANDA_COMPETITION = "prostate-cancer-grade-assessment"
SIIM_COMPETITION = "siim-acr-pneumothorax-segmentation"