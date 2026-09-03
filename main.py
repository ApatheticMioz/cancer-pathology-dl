#!/usr/bin/env python3
"""Unified CLI entry point for the multi-task medical imaging training pipeline.

Consolidates all execution logic into a single script with argparse-based
configuration. Supports three experimental phases (V1, V2, V2.1) and
four datasets (TCGA, PANDA, SIIM, PanNuke) with two encoder backbones.

Usage examples::

    # Run V1 baseline on TCGA with VGG16
    python main.py --phase v1 --datasets tcga --encoders vgg16

    # Run V2 (GradNorm) on all datasets
    python main.py --phase v2

    # Run V2.1 PanNuke control
    python main.py --phase v2.1 --datasets pannuke

    # Dry-run to inspect configuration
    python main.py --phase v2 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.config import (
    CHECKPOINT_DIR,
    DATASET_META,
    DATASET_ROOTS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASETS,
    DEFAULT_ENCODERS,
    EPOCH_LOG_FILE,
    PHASE_CONFIGS,
    PAPER_TARGETS,
    RANDOM_SEED,
    REPRO_DISABLE_CUDNN,
    REPRO_SUMMARY_FILE,
    REQUIRED_MATRIX,
)
from src.data import load_dataset_bundle
from src.training import train_kfold_cv, train_single_run
from src.utils import atomic_json_write, fmt_seconds, now_iso

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with phase-aware defaults."""
    p = argparse.ArgumentParser(
        description="Multi-task UNet training for medical imaging (V1/V2/V2.1 phases)"
    )

    # Experiment selection
    p.add_argument(
        "--phase",
        choices=["v1", "v2", "v2.1"],
        default="v1",
        help="Experimental phase (default: v1)",
    )
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--encoders", nargs="*", default=None)
    p.add_argument("--matrix", choices=["required", "cartesian"], default="required")

    # K-Fold Cross-Validation flags
    p.add_argument(
        "--k-folds",
        type=int,
        default=None,
        help="Number of folds for group-aware cross-validation (e.g., 5 for 5-fold GroupKFold)",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Specific fold index to run (0-indexed) when --k-folds is enabled. If omitted, runs all folds.",
    )

    # Training hyperparameters (overridden by --phase unless explicitly set)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lambda-seg", type=float, default=None)
    p.add_argument("--lambda-cls", type=float, default=None)
    p.add_argument("--gradnorm-alpha", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--cache-size", type=int, default=-1)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)

    # I/O flags
    p.add_argument("--force-redownload-all", action="store_true")
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--dry-run", action="store_true")

    compile_group = p.add_mutually_exclusive_group()
    compile_group.add_argument("--compile", dest="compile", action="store_true")
    compile_group.add_argument("--no-compile", dest="compile", action="store_false")
    p.set_defaults(compile=False)

    # Ablation flags
    p.add_argument(
        "--no-skip-connections",
        action="store_true",
        default=False,
        help="Ablation: bypass UNet decoder skip connections (tests high-res feature transfer)",
    )
    p.add_argument(
        "--no-macenko",
        action="store_true",
        default=False,
        help="Ablation: bypass Macenko normalization (tests domain shift impact)",
    )
    p.add_argument(
        "--enable-gradnorm",
        action="store_true",
        default=False,
        help="Explicitly enable GradNorm, overriding phase default",
    )
    p.add_argument(
        "--disable-gradnorm",
        action="store_true",
        default=False,
        help="Explicitly disable GradNorm, overriding phase default",
    )
    p.add_argument(
        "--static-weights",
        action="store_true",
        default=False,
        help="Use fixed lambda weights for loss computation, bypassing GradNorm dynamic updates",
    )

    # Smoke test flag
    p.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Smoke test mode: forces 1 epoch, 2 batches max, no checkpoint saving",
    )

    # Output path for the summary JSON (enables collision-free parallel runs)
    p.add_argument(
        "--summary-out",
        type=str,
        default=None,
        help="Path for the output summary JSON (default: checkpoints/optimized_summary.json)",
    )

    # Run label for consistent checkpoint/summary naming
    p.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Run label for checkpoint naming (e.g. '01_g1_tcga_vgg16'). "
        "If omitted, auto-generated as '<dataset>_<encoder>'. "
        "Checkpoints are always named ckpt_<run-label>_best_<timestamp>.pth.",
    )

    return p

# ---------------------------------------------------------------------------
# Phase configuration application
# ---------------------------------------------------------------------------

def apply_phase_config(args: argparse.Namespace) -> None:
    """Override CLI defaults with phase-specific hyperparameters.

    Explicitly provided CLI values take precedence over phase defaults.
    """
    phase = PHASE_CONFIGS[args.phase]

    if args.lr is None:
        args.lr = phase["lr"]
    if args.epochs is None:
        args.epochs = phase["epochs"]
    if args.patience is None:
        args.patience = phase["patience"]
    if args.lambda_seg is None:
        args.lambda_seg = phase["lambda_seg"]
    if args.lambda_cls is None:
        args.lambda_cls = phase["lambda_cls"]
    if args.gradnorm_alpha is None:
        args.gradnorm_alpha = phase["gradnorm_alpha"]

    # GradNorm toggle is phase-controlled
    args.use_gradnorm = phase["use_gradnorm"]

    logger.info(
        "Phase %s applied: lr=%g, epochs=%d, patience=%d, "
        "lambda_seg=%g, lambda_cls=%g, gradnorm=%s, alpha=%g",
        args.phase, args.lr, args.epochs, args.patience,
        args.lambda_seg, args.lambda_cls, args.use_gradnorm, args.gradnorm_alpha,
    )

# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------

def _selected_runs(matrix_mode: str, datasets: list[str], encoders: list[str]) -> list[tuple[str, str]]:
    """Build the (dataset, encoder) run matrix."""
    datasets = [d.lower().strip() for d in datasets if d.strip()]
    encoders = [e.strip() for e in encoders if e.strip()]

    unsupported = [d for d in datasets if d not in DATASET_META]
    if unsupported:
        raise ValueError(f"Unsupported datasets: {unsupported}")

    if matrix_mode == "required":
        runs = [(d, e) for d, e in REQUIRED_MATRIX if d in datasets and e in encoders]
    else:
        runs = [(d, e) for d in datasets for e in encoders]

    if not runs:
        raise RuntimeError("No runs selected. Check --datasets/--encoders")
    return runs


def _paper_compare(dataset: str, encoder: str, result: dict) -> dict:
    """Compare run results against paper reference targets."""
    key = f"{dataset}_{encoder}"
    target = PAPER_TARGETS.get(key)
    if not target:
        return {}
    return {
        "paper_acc": float(target["acc"]),
        "paper_dice": float(target["dice"]),
        "acc_delta": float(result["final_val_acc"] - target["acc"]),
        "dice_delta": float(result["final_val_dice"] - target["dice"]),
    }


def _hardware_profile() -> dict:
    """Collect hardware information for audit logging."""
    cpu_affinity = None
    try:
        cpu_affinity = len(os.sched_getaffinity(0))
    except Exception:
        cpu_affinity = None

    cpu_logical = os.cpu_count() or 0
    meminfo = Path("/proc/meminfo")
    ram_total_gb = 0.0
    ram_available_gb = 0.0
    if meminfo.exists():
        try:
            for line in meminfo.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    key = parts[0][:-1]
                    val = float(parts[1])
                    if key == "MemTotal":
                        ram_total_gb = val / (1024.0 * 1024.0)
                    elif key == "MemAvailable":
                        ram_available_gb = val / (1024.0 * 1024.0)
        except Exception:
            pass

    gpu_profile = {"name": None, "total_mem_gb": 0.0, "free_mem_gb": 0.0, "capability": None}
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            gpu_profile = {
                "name": props.name,
                "total_mem_gb": round(float(total_bytes) / (1024.0**3), 2),
                "free_mem_gb": round(float(free_bytes) / (1024.0**3), 2),
                "capability": f"{props.major}.{props.minor}",
            }
        except Exception:
            gpu_profile = {"name": torch.cuda.get_device_name(0), "total_mem_gb": 0.0, "free_mem_gb": 0.0, "capability": None}

    return {
        "cpu_logical": int(cpu_logical),
        "cpu_affinity": int(cpu_affinity) if cpu_affinity is not None else None,
        "ram_total_gb": round(float(ram_total_gb), 2),
        "ram_available_gb": round(float(ram_available_gb), 2),
        "gpu": gpu_profile,
    }


def _resolve_summary_path(args: argparse.Namespace) -> Path:
    """Resolve the target summary file path from CLI args."""
    if args.summary_out:
        p = Path(args.summary_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return REPRO_SUMMARY_FILE


def run_reproduction(args: argparse.Namespace) -> dict:
    """Execute the full training pipeline.

    Sets up device, seeds, hardware profiling, dataset loading,
    and iterates over the (dataset, encoder) run matrix.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = _resolve_summary_path(args)

    # RTX 3090 (Ampere) optimizations
    # TF32: maximize Tensor Core throughput without losing practical precision
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # CuDNN benchmarking: input sizes per dataset are fixed, so CuDNN can profile
    # and select the fastest convolution algorithms.
    torch.backends.cudnn.benchmark = True
    if REPRO_DISABLE_CUDNN:
        torch.backends.cudnn.enabled = False
        logger.info("CuDNN disabled via REPRO_DISABLE_CUDNN for stability")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hardware = _hardware_profile()

    logger.info("Device: %s", device.upper())
    logger.info("Checkpoints: %s", CHECKPOINT_DIR)
    logger.info(
        "Hardware: cpu=%d affinity=%s ram_avail_gb=%.2f gpu=%s gpu_free_gb=%.2f",
        hardware["cpu_logical"], hardware["cpu_affinity"],
        hardware["ram_available_gb"],
        hardware["gpu"]["name"], hardware["gpu"]["free_mem_gb"],
    )

    datasets = args.datasets if args.datasets else DEFAULT_DATASETS
    encoders = args.encoders if args.encoders else DEFAULT_ENCODERS
    runs = _selected_runs(args.matrix, datasets, encoders)

    # Load dataset bundles once per dataset
    bundles = {}
    for dataset in sorted(set(d for d, _ in runs)):
        root = DATASET_ROOTS[dataset]
        bundle = load_dataset_bundle(dataset, root, skip_macenko=args.no_macenko)
        bundles[dataset] = bundle
        labels, counts = np.unique(bundle["labels"], return_counts=True)
        label_dist = {int(k): int(v) for k, v in zip(labels, counts)}
        logger.info(
            "  %s: samples=%d groups=%d labels=%s",
            dataset.upper(), len(bundle["images"]),
            len(np.unique(bundle["groups"])), label_dist,
        )

    if args.dry_run:
        summary = {
            "timestamp": now_iso(),
            "status": "dry-run",
            "phase": args.phase,
            "device": device,
            "runs": [f"{d}_{e}" for d, e in runs],
        }
        atomic_json_write(summary_path, summary)
        logger.info("Dry-run complete. Summary: %s", summary_path)
        return summary

    start_wall = time.time()
    run_results: dict[str, dict] = {}

    for i, (dataset, encoder) in enumerate(runs, start=1):
        run_key = f"{dataset}_{encoder}"
        run_label = args.run_label or run_key

        if args.resume:
            ckpt = CHECKPOINT_DIR / f"ckpt_{run_label}_best.pth"
            if ckpt.exists():
                logger.info("[%d/%d] %s: checkpoint exists; resuming if state available", i, len(runs), run_key)

        if args.k_folds is not None and args.k_folds > 1:
            result = train_kfold_cv(
                dataset=dataset,
                encoder=encoder,
                bundle=bundles[dataset],
                meta=DATASET_META[dataset],
                args=args,
                device=device,
                epoch_log_file=EPOCH_LOG_FILE,
                k_folds=args.k_folds,
                skip_connections=args.no_skip_connections,
                static_weights=getattr(args, "static_weights", False),
                smoke_test=args.smoke_test,
                run_label=run_label,
            )
            # Compare mean metrics to paper target
            result["final_val_acc"] = result["mean_val_acc"]
            result["final_val_dice"] = result["mean_val_dice"]
        else:
            result = train_single_run(
                dataset=dataset,
                encoder=encoder,
                bundle=bundles[dataset],
                meta=DATASET_META[dataset],
                args=args,
                device=device,
                epoch_log_file=EPOCH_LOG_FILE,
                run_index=i,
                total_runs=len(runs),
                skip_connections=args.no_skip_connections,
                static_weights=getattr(args, "static_weights", False),
                smoke_test=args.smoke_test,
                run_label=run_label,
            )
        result.update(_paper_compare(dataset, encoder, result))
        run_results[run_key] = result

        checkpoint_summary = {
            "timestamp": now_iso(),
            "status": "running",
            "phase": args.phase,
            "device": device,
            "hardware": hardware,
            "args": {
                "datasets": datasets,
                "encoders": encoders,
                "phase": args.phase,
                "k_folds": args.k_folds,
                "fold": args.fold,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lambda_seg": args.lambda_seg,
                "lambda_cls": args.lambda_cls,
                "gradnorm_alpha": args.gradnorm_alpha,
                "use_gradnorm": args.use_gradnorm,
                "num_workers": args.num_workers,
                "cache_size": args.cache_size,
                "compile": args.compile,
                "no_skip_connections": args.no_skip_connections,
                "no_macenko": args.no_macenko,
                "seed": args.seed,
            },
            "runs": run_results,
        }
        atomic_json_write(summary_path, checkpoint_summary)

    total_sec = time.time() - start_wall

    # Build results table
    table = []
    for dataset, encoder in runs:
        key = f"{dataset}_{encoder}"
        r = run_results[key]
        if "std_val_acc" in r and "std_val_dice" in r:
            acc_str = f"{100.0 * r['mean_val_acc']:.2f} +/- {100.0 * r['std_val_acc']:.2f}"
            dice_str = f"{100.0 * r['mean_val_dice']:.2f} +/- {100.0 * r['std_val_dice']:.2f}"
        else:
            acc_str = f"{100.0 * r['final_val_acc']:.2f}"
            dice_str = f"{100.0 * r['final_val_dice']:.2f}"

        table.append(
            {
                "dataset": dataset,
                "encoder": encoder,
                "acc": acc_str,
                "dice": dice_str,
                "paper_acc": round(100.0 * r.get("paper_acc", 0.0), 2) if "paper_acc" in r else None,
                "paper_dice": round(100.0 * r.get("paper_dice", 0.0), 2) if "paper_dice" in r else None,
            }
        )

    final_summary = {
        "timestamp": now_iso(),
        "status": "completed",
        "phase": args.phase,
        "device": device,
        "hardware": hardware,
        "duration_sec": float(total_sec),
        "duration_hms": fmt_seconds(total_sec),
        "args": {
            "datasets": datasets,
            "encoders": encoders,
            "phase": args.phase,
            "k_folds": args.k_folds,
            "fold": args.fold,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lambda_seg": args.lambda_seg,
            "lambda_cls": args.lambda_cls,
            "gradnorm_alpha": args.gradnorm_alpha,
            "use_gradnorm": args.use_gradnorm,
            "num_workers": args.num_workers,
            "cache_size": args.cache_size,
            "compile": args.compile,
            "no_skip_connections": args.no_skip_connections,
            "no_macenko": args.no_macenko,
            "seed": args.seed,
        },
        "runs": run_results,
        "table": table,
    }

    atomic_json_write(summary_path, final_summary)

    logger.info("\nFinal metrics")
    logger.info("Dataset  Encoder         Acc(%)             Dice(%)")
    logger.info("-----------------------------------------------------------------")
    for row in table:
        logger.info("%-8s %-14s %-18s %-18s", row["dataset"], row["encoder"], str(row["acc"]), str(row["dice"]))
    logger.info("-----------------------------------------------------------------")
    logger.info("Total wall time: %s", fmt_seconds(total_sec))
    logger.info("Summary: %s", summary_path)

    return final_summary

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry point."""
    # Resolve working directory to project root
    os.chdir(Path(__file__).resolve().parent)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    parser = build_arg_parser()
    args = parser.parse_args()

    # Apply phase-specific defaults
    apply_phase_config(args)

    # GradNorm CLI overrides
    if args.enable_gradnorm:
        args.use_gradnorm = True
        if args.gradnorm_alpha is None or args.gradnorm_alpha == 0.0:
            args.gradnorm_alpha = 1.5
        logger.info("--enable-gradnorm: GradNorm explicitly enabled (phase default overridden, alpha=%g)", args.gradnorm_alpha)
    elif args.disable_gradnorm:
        args.use_gradnorm = False
        logger.info("--disable-gradnorm: GradNorm explicitly disabled (phase default overridden)")

    # GradNorm requires create_graph=True (double-backward), which is incompatible
    # with torch.compile's AOT Autograd.  Force compile off when GradNorm is active.
    if args.use_gradnorm and args.compile:
        logger.warning(
            "GradNorm requires double-backward graph. Disabling torch.compile "
            "for this run to prevent AOT Autograd crashes."
        )
        args.compile = False

    # Set checkpoint directory
    args.checkpoint_dir = CHECKPOINT_DIR

    run_reproduction(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())