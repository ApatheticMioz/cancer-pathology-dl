# Codebase Research & Diagnostic Report

**Generated:** 2026-06-29
**Repository:** Multi-task medical imaging DL pipeline (TCGA, PANDA, SIIM, PanNuke)

---

## 1. High-Level Architecture

The project implements a multi-task U-Net for simultaneous classification and segmentation of cancer pathologies. The pipeline flows as follows:

```
Raw Images → Dataset Parser → GroupShuffleSplit → Macenko Normalization (optional)
    → MultiTaskDataset (LRU cache) → DataLoader (tuned workers/prefetch)
    → MultiTaskUNet (SMP Unet encoder + custom cls head)
    → GradNorm Balancer (dynamic loss weighting) or Static λ weights
    → BCEWithLogitsLoss / CrossEntropyLoss → AMP (bfloat16) + torch.compile
    → Checkpointing (.pth + .state.pt) + JSONL epoch logging
```

**Three experimental phases:**
- **V1** — Baseline: static loss weights (λ_seg=5, λ_cls=1), lr=1e-3, no GradNorm, no Macenko
- **V2** — Enhanced: GradNorm (α=1.5), Macenko ON, lr=1e-4
- **V2.1** — PanNuke control: V1 pipeline config on PanNuke

**26-run experimental matrix** organized in 5 groups: Naked Baseline (6), Final Form (6), PanNuke Crucible (4), Optimization Teardown (6), Ablations (4).

---

## 2. Directory Map

| Path | Contents | Assessment |
|---|---|---|
| `src/` | Core Python package (7 files) | Well-structured; see §4 for issues |
| `main.py` | Unified CLI entry point | Good; single source of truth |
| `run_all_experiments.sh` | Master 26-run orchestrator | Production-quality bash |
| `run_remaining.sh` | Re-run 6 failed/incomplete runs | Legacy; superseded by recovery script |
| `run_macenko_recovery.sh` | Macenko data corruption recovery | Legacy; 7 corrupted runs archived |
| `run_smoke_test.sh` | 10-test smoke test suite | Good; validates all flag combos |
| `update_env.sh` | PyTorch version check + upgrade | Useful but narrow scope |
| `data/` | 4 dataset roots (PANDA, PanNuke, SIIM, TCGA) | Expected |
| `checkpoints/` | 8 `.pth` weights, 8 `.state.pt`, 26 summary JSONs, 1 JSONL log | All 26 runs completed |
| `logs/` | 27 `.log` files (one duplicate: `run_09_g2_panda_v2_panda_vgg16.log`) | Minor naming collision |
| `smoke_test_logs/` | 10 smoke test logs | Clean |
| `smoke_test.log` | Legacy concatenated smoke log (346+ lines of torch.compile autotune spam) | Noisy; contains evidence of a bug (see §4) |
| `paper/` | LaTeX drafts, CSV results, bibliography, figures | Mixed state; multiple draft versions |
| `archive/` | `baseline_repro/`, `corrupted_runs/`, `old_checkpoints_v2/`, `research_drafts/` | Properly quarantined |
| `docs/` | Single paper markdown copy | Redundant with `paper/` |
| `venv/` | Python virtual environment | Standard |

---

## 3. Root File Audit

### `main.py` (502 lines) — **KEEP, PRIMARY ENTRY POINT**
Unified argparse CLI. Handles phase config, hardware profiling, device setup, dataset loading, run matrix iteration, paper comparison, and summary JSON output. Well-documented. Supports `--dry-run`, `--smoke-test`, `--resume`, and all ablation flags.

### `run_all_experiments.sh` (203 lines) — **KEEP, MASTER ORCHESTRATOR**
Launches all 26 runs with MAX_JOBS=3 concurrency. Each run gets a unique log and summary file. Groups runs sequentially (waits for group completion before next group). Clean bash with proper error handling.

### `run_remaining.sh` (221 lines) — **OBSOLETE**
Re-runs 6 failed runs (9, 10, 23-26) from the original execution. Includes Phase 1 to resolve PANDA Macenko data dependency via symlink. All 6 runs have since completed successfully (summary JSONs exist in `checkpoints/`). This script is no longer needed.

### `run_macenko_recovery.sh` (194 lines) — **OBSOLETE**
Recovery script for 7 runs corrupted by bad Macenko data. Archives old logs/summaries to `archive/corrupted_runs/`, then re-runs. All 7 runs have been re-run and completed. The `archive/corrupted_runs/` directory confirms this script already executed successfully.

### `run_smoke_test.sh` (167 lines) — **KEEP**
10-test matrix covering all phase/encoder/ablation combinations. Uses `--smoke-test` flag (1 epoch, 2 batches, no checkpoint save). Color-coded output with per-test log files. Well-structured.

### `update_env.sh` (83 lines) — **KEEP (LOW PRIORITY)**
Checks PyTorch version, upgrades if <2.0. CUDA-aware. Narrow but useful for onboarding.

### `smoke_test.log` — **CLEANUP CANDIDATE**
Legacy concatenated log. Contains 346+ lines of `torch.compile` autotune benchmarking output. Contains evidence of a `ValueError` bug in the smoke test code path (see §4). Should be regenerated or removed.

### Overlap Analysis
- `run_remaining.sh` and `run_macenko_recovery.sh` share ~80 lines of identical concurrency infrastructure (`launch_job`, `count_active`, `wait_for_slot`, `wait_all`, `ts`). This boilerplate is duplicated across all 3 shell scripts (`run_all_experiments.sh`, `run_remaining.sh`, `run_macenko_recovery.sh`).

---

## 4. Technical Debt & Refactoring Targets

### Critical Bugs

**B1 — Smoke test unpacking bug (`src/modeling.py:773`)**
The `smoke_test.log` shows `ValueError: not enough values to unpack (expected 3, got 2)` at the line `_, final_acc, final_dice = vl_acc, vl_dice`. In smoke test mode, the code assigns `final_acc, final_dice = vl_acc, vl_dice` (2 values) but the unpacking pattern expects 3. The current `modeling.py` shows this as `final_acc, final_dice = vl_acc, vl_dice` which is correct for 2 values — this bug was already fixed between the log capture and current code. However, the `smoke_test.log` is stale evidence of the old broken state.

**B2 — CSV parser splits `mobilenet_v2` incorrectly (`src/aggregate_results.py:266`)**
The `paper_results_matrix.csv` shows corrupted rows where `mobilenet_v2` is split: Dataset becomes `TCGA_MOBILENET`, Encoder becomes `v2`. The `rsplit("_", 1)` at line 266 splits `"tcga_mobilenet_v2"` into `("tcga_mobilenet", "v2")`. This breaks all MobileNetV2 rows in the output CSV and LaTeX table.

**B3 — Duplicate log file (`logs/run_09_g2_panda_v2_panda_vgg16.log`)**
A naming collision produced a malformed log filename. Likely from a re-run with a different naming convention. Should be cleaned up.

### Architectural Debt

**D1 — Shell script boilerplate duplication**
The concurrency infrastructure (`launch_job`, `count_active`, `wait_for_slot`, `wait_all`, `ts`) is copy-pasted identically across 3 shell scripts (~80 lines × 3 = 240 lines of duplicated code). Extract to a shared `scripts/lib.sh` or similar.

**D2 — `modeling.py` is 873 lines — too monolithic**
This single file contains:
- System helpers (CPU/RAM detection) — 30 lines
- GradNorm class — 40 lines
- MultiTaskUNet model — 60 lines
- Dice metric — 35 lines
- Training epoch loop — 130 lines
- Checkpoint helpers — 60 lines
- DataLoader tuning — 40 lines
- `train_single_run` orchestrator — 360 lines

Should be split into: `models.py` (UNet + GradNorm), `metrics.py` (Dice), `training.py` (epoch loop + train_single_run), `checkpoints.py` (save/load), and `loader_tuning.py`.

**D3 — Hardcoded environment flags scattered across files**
`REPRO_DISABLE_CUDNN`, `REPRO_STRICT_BATCH_CHECKS`, `REPRO_ALLOW_BIG_CACHE`, `REPRO_ALLOW_UNC_WORKERS`, `REPRO_TORCH_COMPILE_BACKEND` — 5 environment variables referenced in `main.py` and `modeling.py` with no central registry or documentation.

**D4 — `recover_forensics.py` is a dead-end artifact**
This script targets `checkpoints_v2/epoch_log.jsonl` (a path that no longer exists — checkpoints are now in `checkpoints/`). It hardcodes a 20-run sequence that doesn't match the current 26-run matrix. The output files (`paper/recovered_results_matrix.csv`, `paper/recovered_latex_table.txt`) are stale. This script should be archived or deleted.

**D5 — `ISIC_URLS` and `ISIC_KAGGLE_FALLBACK_REFS` in `config.py` are dead code**
Lines 138-153 define ISIC 2018 download URLs and Kaggle fallback references, but no code in the repository actually uses these constants. The ISIC dataset is not among the 4 supported datasets.

**D6 — Checkpoint naming collision risk**
`checkpoints/` stores weights as `{dataset}_{encoder}_best.pth`. With 4 datasets × 2 encoders = 8 files, this works. But if a new dataset or encoder is added, or if parallel runs write to the same directory, collisions occur. The summary JSONs use unique names (`summary_XX_name.json`), but the weight files do not.

**D7 — `paper/` directory is disorganized**
Contains 17 files mixing LaTeX source (`.tex`, `.aux`, `.log`, `.out`), PDFs, markdown drafts, CSV results, and a `bibliography/` and `figures/` subdirectory. Multiple draft versions coexist (`full_draft.tex`, `main.tex`, `output_humanized.tex`). Build artifacts (`.aux`, `.log`) should be cleaned.

### Modularity Assessment

| Component | Location | Separation Quality | Notes |
|---|---|---|---|
| GradNorm | `src/modeling.py` | Poor | Mixed with model arch and training loop |
| Macenko | `src/apply_macenko.py` | Good | Self-contained, multiprocessing-ready |
| Metrics (Dice) | `src/modeling.py` | Poor | Buried in modeling module |
| Data loading | `src/data.py` | Good | Clean separation of parsers, transforms, splits |
| Config | `src/config.py` | Good | Centralized, well-structured |
| Utils | `src/utils.py` | Good | Small, focused, atomic I/O |
| Results aggregation | `src/aggregate_results.py` | Good | But has the `rsplit` bug (B2) |
| Training loop | `src/modeling.py` | Poor | 873-line file does too much |

---

## 5. Data Flow Map

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATASET PARSING (src/data.py)                                │
│                                                                 │
│   TCGA:     root/<patient>/*.tif + *_mask.tif → binary label    │
│   PANDA:    train.csv + images/ + train_label_masks/ → ISUP 0-5 │
│   SIIM:     preprocessed/index.csv → absolute path resolution   │
│   PanNuke:  preprocessed/index.csv → absolute path resolution   │
│                                                                 │
│   Output: {images[], masks[], labels[], groups[]}               │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. GROUP-AWARE SPLIT (src/data.py:make_group_split)             │
│                                                                 │
│   If unique(groups) == len(samples) → StratifiedShuffleSplit    │
│   Else → GroupShuffleSplit (prevents patient-level leakage)     │
│   80/20 train/val split, seed=42                                │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. MACENKO NORMALIZATION (src/apply_macenko.py)                 │
│                                                                 │
│   Offline, pre-computed. Gold standard reference from MEDIAN    │
│   stain matrix of ALL samples (excluding >50% white images).    │
│   Output: preprocessed_macenko_fixed/images/                    │
│   Symlinked to: preprocessed_macenko/images/                    │
│   Applied to: PANDA, PanNuke only (TCGA, SIIM skip)             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. MULTI-TASK DATASET (src/data.py:MultiTaskDataset)            │
│                                                                 │
│   LRU cache (configurable size). Loads (image, mask) pairs.     │
│   Binary masks: threshold → float. Multi-class: clip → int64.   │
│   Transforms: Albumentations (flip, rotate, affine, normalize)  │
│   Output: (image_tensor, mask_tensor, label_tensor)             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. DATA LOADER (src/modeling.py:_initial_loader_tuning)         │
│                                                                 │
│   Auto-tunes workers, prefetch_factor, persistent_workers       │
│   based on dataset type, RAM, and CPU budget.                   │
│   WSL UNC path detection → forces workers=0 if needed           │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. MODEL (src/modeling.py:MultiTaskUNet)                        │
│                                                                 │
│   SMP Unet (VGG16 or MobileNetV2 encoder, ImageNet weights)     │
│   Shared encoder → decoder + segmentation head                  │
│   Shared encoder bottleneck → AdaptiveAvgPool → MLP → cls head  │
│   Ablation: skip connections zeroed via --no-skip-connections   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. LOSS & GRADNORM (src/modeling.py:GradNormBalancer)           │
│                                                                 │
│   Seg: BCEWithLogitsLoss (binary) or CrossEntropyLoss (multi)   │
│   Cls: CrossEntropyLoss with inverse-frequency class weights    │
│   V1: Static λ (5×seg + 1×cls)                                 │
│   V2: GradNorm — learnable log_weights, gradient magnitude      │
│        balancing on shared encoder params, α-asymmetric target   │
│   AMP: bfloat16 autocast + GradScaler                           │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. CHECKPOINTING + LOGGING                                      │
│                                                                 │
│   Best model: {dataset}_{encoder}_best.pth                      │
│   Full state: {dataset}_{encoder}_best.state.pt                 │
│   Epoch log: checkpoints/epoch_log.jsonl (append-only JSONL)    │
│   Summary: checkpoints/summary_XX_name.json (atomic write)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Summary of Findings

| Category | Count | Severity |
|---|---|---|
| Critical bugs | 2 (B1 fixed, B2 active) | HIGH |
| Obsolete scripts | 2 (`run_remaining.sh`, `run_macenko_recovery.sh`) | LOW |
| Dead code | 2 (`recover_forensics.py`, `ISIC_URLS`) | MEDIUM |
| Monolithic files | 1 (`modeling.py` at 873 lines) | MEDIUM |
| Duplicated boilerplate | 1 (shell concurrency, 240 lines) | LOW |
| Disorganized directories | 1 (`paper/`) | LOW |
| Stale artifacts | 2 (`smoke_test.log`, `recover_forensics.py` outputs) | LOW |

**Overall assessment:** The `/src/` package is well-designed at the module level (`config.py`, `data.py`, `utils.py`, `apply_macenko.py` are clean and focused). The primary architectural issue is `modeling.py` being a catch-all for models, metrics, training, and checkpointing. The root-level shell scripts show the evolution of a research project: the master orchestrator is production-quality, but the recovery and re-run scripts are now obsolete artifacts of past failures. The CSV aggregation bug (B2) is the most impactful active issue, as it corrupts the paper-ready results table for all MobileNetV2 runs.