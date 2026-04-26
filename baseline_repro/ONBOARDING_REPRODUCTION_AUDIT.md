# Onboarding Guide: Reproduction and Scientific Audit

## 1) Project Overview

This repository is a literal, implementation-level reproduction of the paper:

- Multi-Task Deep Learning for Simultaneous Classification and Segmentation of Cancer Pathologies in Diverse Medical Imaging Modalities (Onco 2025)

The codebase has two goals:

1. Reproduce the published multi-task setup as faithfully as possible from stated methods.
2. Establish realistic, leakage-resistant baselines and document where the paper's reported numbers are not methodologically reproducible.

In practice, this means the project executes the exact architecture family and hyperparameters declared in the manuscript, then audits whether those claims hold under strict data hygiene (for example patient-level grouping where required) and explicit task definitions.

---

## 2) Quickstart / Execution

### Environment and run command

The latest successful run was:

```bash
cd /home/apath/Work/temp/final && source venv/bin/activate && python train.py --datasets tcga panda siim --encoders vgg16 mobilenet_v2 --matrix required --epochs 50 --patience 10 --batch-size 32 --lr 0.001 --lambda-seg 5 --lambda-cls 1 --num-workers -1 --cache-size -1 --no-resume --no-compile
```

### What this command does

- Runs the required paper matrix for datasets {tcga, panda, siim} x encoders {vgg16, mobilenet_v2}.
- Uses paper-aligned optimization settings:
  - Epochs: 50
  - Batch size: 32
  - Adam lr: 0.001
  - Multi-task loss: lambda_seg=5, lambda_cls=1
- Enables automatic host tuning for DataLoader workers and cache (with explicit paper batch size preserved).
- Forces a fresh training run (`--no-resume`) and disables torch compile acceleration (`--no-compile`).

### CLI parameters (operational meaning)

- `--datasets`: Dataset keys to run. Supported in code: `tcga`, `isic`, `panda`, `siim`.
- `--encoders`: Encoder backbones passed to segmentation_models_pytorch UNet (`vgg16`, `mobilenet_v2`).
- `--matrix`: `required` uses predefined paper matrix; `cartesian` runs full product of selected datasets x encoders.
- `--epochs`: Max epochs.
- `--patience`: Early stopping patience on validation objective.
- `--batch-size`: Batch size (paper value is 32).
- `--lr`: Adam learning rate.
- `--lambda-seg`, `--lambda-cls`: Weights for total loss = lambda_seg * seg_loss + lambda_cls * cls_loss.
- `--num-workers`: `<0` enables hardware/dataset auto-tuning.
- `--cache-size`: `<0` enables auto cache policy, with safety guards for large datasets.
- `--resume` / `--no-resume`: Resume from `*.state.pt` training state or force fresh run.
- `--compile` / `--no-compile`: Enable/disable torch.compile wrapper.
- `--dry-run`: Build plan/audit without training.

---

## 3) Codebase Architecture

### Entry point

- `train.py`
  - Sets working directory to repository root.
  - Builds CLI parser from `repro.runner`.
  - Forces checkpoint directory from `repro.config.CHECKPOINT_DIR`.
  - Calls `run_reproduction(args)`.

### Orchestration and experiment control

- `repro/runner.py`
  - Parses run matrix (`required` vs `cartesian`).
  - Applies deterministic seeds.
  - Profiles hardware and prints runtime profile.
  - Calls dataset preparation (`prepare_datasets`) and writes dataset audit.
  - Loads each dataset once, then reuses parsed bundles across encoder runs.
  - Executes training via `train_single_run`.
  - Compares each run against paper targets (`paper_acc`, `paper_dice`, and deltas).
  - Writes machine-readable outputs:
    - `checkpoints/reproduction_summary.json`
    - `checkpoints/epoch_log.jsonl`

### Data ingestion and split logic

- `repro/data.py`
  - Parsing functions:
    - `parse_tcga`: reads image/mask pairs by patient directory and sets groups to patient ID.
    - `parse_panda`: uses `train.csv` (`image_id`, `isup_grade`) and paired masks.
    - `parse_siim`: consumes preprocessed index with explicit image/mask/label/group columns.
    - `parse_isic`: supported but not part of the latest required run.
  - Split strategy:
    - `make_group_split` uses group-aware split when groups are not unique.
    - Falls back to stratified split only when every sample is effectively its own group.
  - Dataset wrapper:
    - `MultiTaskDataset` returns image, segmentation target, and classification label.
    - Binary segmentation datasets use thresholded masks.
    - PANDA keeps multi-class segmentation target (`seg_classes=6`).

### Model and training loop

- `repro/modeling.py`
  - `MultiTaskUNet`:
    - Shared UNet encoder-decoder for segmentation.
    - Classification head on encoder bottleneck (adaptive pool + MLP).
  - Losses:
    - Segmentation: BCEWithLogits for binary datasets, CrossEntropy for multi-class segmentation.
    - Classification: weighted CrossEntropy (class-balanced weights from training distribution).
  - Total objective:
    - `loss = lambda_seg * seg_loss + lambda_cls * cls_loss`
  - Checkpointing:
    - Best checkpoint selected by validation joint loss.
    - Saves model weights (`*_best.pth`) and resumable optimizer/state (`*_best.state.pt`).

### Dataset preparation and audit

- `repro/prepare.py`
  - Verifies dataset roots and expected files.
  - Handles downloads/extractions/preprocessing (including SIIM preprocessing index expectations).
  - Produces `checkpoints/dataset_audit.json` used as a reproducibility artifact.

### Utility helpers

- `repro/utils.py`
  - Atomic JSON writes, JSONL appenders, timestamp and duration helpers.

---

## 4) Fidelity and Methodology

This implementation intentionally follows the paper's explicit setup:

- Encoders: VGG16 and MobileNetV2.
- Multi-task objective weights: lambda_seg = 5, lambda_cls = 1.
- Training schedule: 50 epochs, batch size 32, Adam optimizer, lr = 0.001.
- Joint classification + segmentation architecture using shared UNet encoder features and a classification head.

Methodological rigor added by this repo:

- Group-aware splitting for datasets with meaningful grouping (for example patient-level TCGA grouping).
- Explicit, auditable class definitions in code (including PANDA 6-class setup).
- Run artifacts persisted as machine-readable JSON/JSONL for audit trails.

---

## 5) The Audit / Critical Findings

This section documents the main discrepancies between paper claims and leakage-resistant reproduction.

### A) TCGA-LGG (Brain Tumor): data leakage in slice-level random split

Paper claim:

- Around 98% Dice and 89-90% classification accuracy.

Audit finding:

- Pooling all 3929 2D slices and random image-level splitting leaks near-duplicate volumetric slices from the same patient across train/test.
- This repository uses patient-level grouping for split boundaries, forcing generalization to unseen brains.
- Under patient-level evaluation, Dice is far below 98% and becomes a realistic baseline rather than a leakage-inflated number.

Observed (latest run):

- TCGA VGG16: 85.22% accuracy, 75.35% Dice.
- TCGA MobileNetV2: 93.96% accuracy, 86.72% Dice.

Interpretation:

- Accuracy can remain high while overlap quality (Dice) drops significantly under proper grouping.
- The paper Dice claims are not supported under strict patient-level separation.

### B) PANDA (Prostate): undocumented class collapse (hidden binarization)

Paper description:

- Prostate task described as 6 classes (Background, Stroma, Benign epithelium, Gleason 3, 4, 5).

Audit finding:

- Paper-level high scores are consistent with an undocumented binary collapse (Tumor vs Background), not the stated 6-class task.
- This repository keeps PANDA as an actual 6-class classification + segmentation problem.

Observed (latest run):

- PANDA VGG16: 41.06% accuracy, 39.92% Dice.
- PANDA MobileNetV2: 43.68% accuracy, 40.23% Dice.

Interpretation:

- Values around ~40-43% are expected for the true 6-class formulation and are incompatible with reported ~88% paper accuracy unless task definition is changed.

### C) SIIM (Pneumothorax): unrealistic 99% Dice baseline

Paper claim:

- 99% Dice on SIIM pneumothorax segmentation.

Audit finding:

- For faint boundary 2D chest X-ray segmentation with standard UNet-style setup, a true 99% Dice is clinically and computationally implausible.
- The observed gap strongly suggests split leakage (same pattern as random image-level partitioning).

Observed (latest run):

- SIIM VGG16: 77.70% accuracy, 77.74% Dice.
- SIIM MobileNetV2: 79.39% accuracy, 77.74% Dice.

Interpretation:

- The reproduced SIIM Dice (~77.7%) is a realistic baseline under strict evaluation.

---

## 6) Best Outputs and How to Read Them

Primary artifacts are in `checkpoints/`:

- `reproduction_summary.json`
  - Canonical run summary.
  - Contains per-run metrics and paper deltas:
    - `final_val_acc`, `final_val_dice`
    - `paper_acc`, `paper_dice`
    - `acc_delta`, `dice_delta`
- `dataset_audit.json`
  - Snapshot of dataset readiness, counts, and resolved paths used for run integrity.
- `epoch_log.jsonl`
  - Epoch-by-epoch training/validation metrics for all runs.
- `*_best.pth`
  - Best model weights per dataset/encoder.
- `*_best.state.pt`
  - Resumable training state (model + optimizer + early stopping state).

### Quick interpretation workflow

1. Open `checkpoints/reproduction_summary.json`.
2. Compare each run's `final_val_acc` and `final_val_dice` against `paper_acc` and `paper_dice`.
3. Use `acc_delta` / `dice_delta` to quantify inflation or drop.
4. Verify data integrity assumptions in `checkpoints/dataset_audit.json`.
5. If needed, inspect training dynamics in `checkpoints/epoch_log.jsonl`.

### True baseline snapshot (latest successful run)

| Dataset | Encoder | Accuracy | Dice | Paper Accuracy | Paper Dice |
|---|---|---:|---:|---:|---:|
| TCGA | vgg16 | 85.22% | 75.35% | 89.00% | 97.00% |
| TCGA | mobilenet_v2 | 93.96% | 86.72% | 90.00% | 98.00% |
| PANDA | vgg16 | 41.06% | 39.92% | 87.00% | 98.00% |
| PANDA | mobilenet_v2 | 43.68% | 40.23% | 88.00% | 99.00% |
| SIIM | vgg16 | 77.70% | 77.74% | 82.00% | 99.00% |
| SIIM | mobilenet_v2 | 79.39% | 77.74% | 87.00% | 99.00% |

These values are the operational baseline for this repository under strict, auditable methodology.
