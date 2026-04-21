from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from repro.config import (
    CHECKPOINT_DIR,
    DATASET_META,
    DATASET_ROOTS,
    DEFAULT_DATASETS,
    DEFAULT_ENCODERS,
    EPOCH_LOG_FILE,
    PAPER_TARGETS,
    RANDOM_SEED,
    REPRO_SUMMARY_FILE,
    REQUIRED_MATRIX,
)
from repro.data import load_dataset_bundle
from repro.modeling import train_single_run
from repro.prepare import prepare_datasets
from repro.utils import atomic_json_write, fmt_seconds, now_iso


def _load_existing_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _normalize_existing_runs(summary: dict) -> dict[str, dict]:
    """Coerce historical summary formats into a run_key -> run_record mapping."""
    if not isinstance(summary, dict):
        return {}

    raw_runs = summary.get("runs", {})
    if isinstance(raw_runs, dict):
        return {str(k): v for k, v in raw_runs.items() if isinstance(v, dict)}

    if isinstance(raw_runs, list):
        normalized: dict[str, dict] = {}
        for item in raw_runs:
            if isinstance(item, str):
                # Dry-run summaries can store only run keys.
                normalized[item] = {"status": "planned"}
                continue

            if isinstance(item, dict):
                key = item.get("run_key") or item.get("key") or item.get("name")
                if isinstance(key, str) and key:
                    normalized[key] = item
        return normalized

    return {}


def _selected_runs(matrix_mode: str, datasets: list[str], encoders: list[str]) -> list[tuple[str, str]]:
    datasets = [d.lower().strip() for d in datasets if d.strip()]
    encoders = [e.strip() for e in encoders if e.strip()]

    unsupported_datasets = [d for d in datasets if d not in DATASET_META]
    if unsupported_datasets:
        raise ValueError(f"Unsupported datasets: {unsupported_datasets}")

    if matrix_mode == "required":
        runs = [(d, e) for (d, e) in REQUIRED_MATRIX if d in datasets and e in encoders]
    else:
        runs = [(d, e) for d in datasets for e in encoders]

    if not runs:
        raise RuntimeError("No runs selected. Check --datasets/--encoders")
    return runs


def _paper_compare(dataset: str, encoder: str, result: dict) -> dict:
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


def run_reproduction(args) -> dict:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")

    datasets = args.datasets if args.datasets else DEFAULT_DATASETS
    encoders = args.encoders if args.encoders else DEFAULT_ENCODERS
    runs = _selected_runs(args.matrix, datasets, encoders)

    # Hard rule: missing/incomplete datasets are purged and redownloaded.
    print("Preparing datasets...")
    audit = prepare_datasets(datasets=sorted(set(datasets)), force_redownload_all=args.force_redownload_all)
    print("Dataset preparation completed.")

    # Parse each dataset once and reuse across encoders.
    bundles = {}
    for dataset in sorted(set(d for d, _ in runs)):
        root = DATASET_ROOTS[dataset]
        bundle = load_dataset_bundle(dataset, root)
        bundles[dataset] = bundle
        labels, counts = np.unique(bundle["labels"], return_counts=True)
        label_dist = {int(k): int(v) for k, v in zip(labels, counts)}
        print(
            f"  {dataset.upper()}: samples={len(bundle['images'])} "
            f"groups={len(np.unique(bundle['groups']))} labels={label_dist}"
        )

    if args.dry_run:
        summary = {
            "timestamp": now_iso(),
            "status": "dry-run",
            "device": device,
            "runs": [f"{d}_{e}" for d, e in runs],
            "dataset_audit": audit,
        }
        atomic_json_write(REPRO_SUMMARY_FILE, summary)
        return summary

    existing = _load_existing_summary(REPRO_SUMMARY_FILE) if args.resume else {}
    existing_runs = _normalize_existing_runs(existing)

    start_wall = time.time()
    run_results: dict[str, dict] = {}

    for i, (dataset, encoder) in enumerate(runs, start=1):
        run_key = f"{dataset}_{encoder}"
        ckpt = CHECKPOINT_DIR / f"{dataset}_{encoder}_best.pth"

        existing_entry = existing_runs.get(run_key)
        if args.resume and isinstance(existing_entry, dict) and existing_entry.get("status") == "completed" and ckpt.exists():
            print(f"[{i}/{len(runs)}] {run_key}: already completed, skipping")
            run_results[run_key] = existing_entry
            continue

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
        )
        result.update(_paper_compare(dataset, encoder, result))
        run_results[run_key] = result

        checkpoint_summary = {
            "timestamp": now_iso(),
            "status": "running",
            "device": device,
            "args": {
                "datasets": datasets,
                "encoders": encoders,
                "matrix": args.matrix,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lambda_seg": args.lambda_seg,
                "lambda_cls": args.lambda_cls,
                "num_workers": args.num_workers,
                "cache_size": args.cache_size,
                "compile": args.compile,
                "seed": args.seed,
                "force_redownload_all": args.force_redownload_all,
            },
            "dataset_audit": audit,
            "runs": run_results,
        }
        atomic_json_write(REPRO_SUMMARY_FILE, checkpoint_summary)

    total_sec = time.time() - start_wall

    table = []
    for dataset, encoder in runs:
        key = f"{dataset}_{encoder}"
        r = run_results[key]
        table.append(
            {
                "dataset": dataset,
                "encoder": encoder,
                "acc": round(100.0 * r["final_val_acc"], 2),
                "dice": round(100.0 * r["final_val_dice"], 2),
                "paper_acc": round(100.0 * r.get("paper_acc", 0.0), 2) if "paper_acc" in r else None,
                "paper_dice": round(100.0 * r.get("paper_dice", 0.0), 2) if "paper_dice" in r else None,
            }
        )

    final_summary = {
        "timestamp": now_iso(),
        "status": "completed",
        "device": device,
        "duration_sec": float(total_sec),
        "duration_hms": fmt_seconds(total_sec),
        "args": {
            "datasets": datasets,
            "encoders": encoders,
            "matrix": args.matrix,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lambda_seg": args.lambda_seg,
            "lambda_cls": args.lambda_cls,
            "num_workers": args.num_workers,
            "cache_size": args.cache_size,
            "compile": args.compile,
            "seed": args.seed,
            "force_redownload_all": args.force_redownload_all,
        },
        "dataset_audit": audit,
        "runs": run_results,
        "table": table,
    }

    atomic_json_write(REPRO_SUMMARY_FILE, final_summary)

    print("\nFinal metrics")
    print("Dataset  Encoder         Acc(%)  Dice(%)")
    print("-----------------------------------------")
    for row in table:
        print(f"{row['dataset']:<8} {row['encoder']:<14} {row['acc']:>6.2f}   {row['dice']:>6.2f}")
    print("-----------------------------------------")
    print(f"Total wall time: {fmt_seconds(total_sec)}")
    print(f"Summary: {REPRO_SUMMARY_FILE}")

    return final_summary


def build_arg_parser():
    import argparse

    cpu_count = os.cpu_count() or 8
    default_workers = max(2, min(16, cpu_count - 2))

    p = argparse.ArgumentParser(description="Rhanoui et al. 2025 multi-task UNet reproduction")
    p.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    p.add_argument("--encoders", nargs="*", default=DEFAULT_ENCODERS)
    p.add_argument("--matrix", choices=["required", "cartesian"], default="required")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=0, help="0 = auto per dataset/encoder")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-seg", type=float, default=5.0)
    p.add_argument("--lambda-cls", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=default_workers)
    p.add_argument("--cache-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)

    p.add_argument("--force-redownload-all", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--dry-run", action="store_true")

    compile_group = p.add_mutually_exclusive_group()
    compile_group.add_argument("--compile", dest="compile", action="store_true")
    compile_group.add_argument("--no-compile", dest="compile", action="store_false")
    p.set_defaults(compile=True)

    return p
