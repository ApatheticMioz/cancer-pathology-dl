from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LEGACY_CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR = BASE_DIR / "checkpoints_v2"
DATASET_AUDIT_FILE = CHECKPOINT_DIR / "dataset_audit.json"
EPOCH_LOG_FILE = CHECKPOINT_DIR / "epoch_log.jsonl"
REPRO_SUMMARY_FILE = CHECKPOINT_DIR / "optimized_summary.json"

DEFAULT_BATCH_SIZE = 32

DATASET_ROOTS = {
    "tcga": BASE_DIR / "TCGA",
    "panda": BASE_DIR / "PANDA_raw",
    "siim": BASE_DIR / "SIIM_raw",
    "pannuke": BASE_DIR / "pannuke",
}

DATASET_META = {
    "tcga": {"num_classes": 2, "img_size": 256, "seg_classes": 1, "binary_positive_min": 1, "use_macenko": False},
    # Paper tables report six prostate grading categories for both tasks.
    "panda": {"num_classes": 6, "img_size": 128, "seg_classes": 6, "use_macenko": True},
    "siim": {"num_classes": 2, "img_size": 224, "seg_classes": 1, "binary_positive_min": 1, "use_macenko": False},
    "pannuke": {"num_classes": 19, "img_size": 256, "seg_classes": 6, "use_macenko": True},
}

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

DEFAULT_ENCODERS = ["vgg16", "mobilenet_v2"]
DEFAULT_DATASETS = ["tcga", "panda", "siim", "pannuke"]

# Paper table values for quick side-by-side comparison.
PAPER_TARGETS = {
    "tcga_vgg16": {"acc": 0.89, "dice": 0.97},
    "tcga_mobilenet_v2": {"acc": 0.90, "dice": 0.98},
    "panda_vgg16": {"acc": 0.87, "dice": 0.98},
    "siim_vgg16": {"acc": 0.82, "dice": 0.99},
    "panda_mobilenet_v2": {"acc": 0.88, "dice": 0.99},
    "siim_mobilenet_v2": {"acc": 0.87, "dice": 0.99},
}

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

RANDOM_SEED = 42
