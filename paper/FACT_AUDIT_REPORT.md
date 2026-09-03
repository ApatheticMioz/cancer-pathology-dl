# Comprehensive Fact Audit Report: Multi-Task Cancer Pathology Study

**Target Repository**: `/home/apath/Work/temp/final`  
**Supervised Session**: `e6b36152-88bf-4a4d-aecc-fdbc6057d073`  
**Worker Session**: `cancer_paper_audit` (`20260903_1`)  
**Audit Status**: 35 Tool Invocations Completed | Ground Truth Verified from Code, Checkpoints, and Data

---

## Executive Summary

This repository is a rigorous **reproduction, diagnostic, and refutation study** evaluating the multi-task learning claims of Rhanoui et al. (*Onco*, 2025: *"Multi-Task Deep Learning for Simultaneous Classification and Segmentation of Cancer Pathologies in Diverse Medical Imaging Modalities"*, DOI: 10.3390/onco5010034).

While the experimental codebase has completed a 26-run benchmark matrix across PANDA, PanNuke, SIIM, and TCGA datasets, this audit identified **critical discrepancies between the draft manuscript ([`paper/main.tex`](file:///home/apath/Work/temp/final/paper/main.tex)) and the actual underlying code/data** that must be resolved for submission to a top-tier medical AI venue (e.g., MICCAI / IEEE TMI / Lancet Digital Health).

---

## 1. Core Clinical & ML Problem Formulation

- **Clinical Task**: Multi-task learning (MTL) performing simultaneous disease grading/classification and lesion/tissue segmentation across diverse imaging modalities (histopathology WSI patches, radiology).
- **Core Hypothesis Evaluated**: Does simultaneous multi-task training with dynamic gradient balancing (GradNorm) and Macenko stain normalization outperform single-task models, or does negative task transfer degrade representation quality?
- **Reproduction Finding**: The study empirically refutes Rhanoui et al.'s reported performance leaps, demonstrating that naive joint segmentation and classification introduces gradient conflict, causing Dice score degradation unless isolated or properly regularized.

---

## 2. Dataset & Preprocessing Pipeline Verification

| Dataset | Modality / Target | Verified Code Setting (`src/config.py`, `src/data.py`) | Manuscript Draft Discrepancy ([`paper/main.tex`](file:///home/apath/Work/temp/final/paper/main.tex)) |
|---|---|---|---|
| **TCGA** | Lung Histopathology (LUAD/LUSC) | $N=3,929$ ($3,151$ train, $778$ validation), 2 classes, $256 \times 256$ | 🔴 **Major Error**: Draft §3.1 describes TCGA as *"TCGA-LGG brain tumor (2D MRI)"* with *"110 patients"*. |
| **PANDA** | Prostate Needle Biopsy (Gleason Grading) | 6-class classification (ISUP $0\text{--}5$), **6-class segmentation** (`seg_classes=6`), $128 \times 128$ | 🟠 **Discrepancy**: Draft claims binary *"epithelial gland segmentation"*. Group splitting is at biopsy image level (`groups.append(image_id)`), not patient level. |
| **PanNuke** | Multi-Tissue Histopathology | 19-class tissue type, 6-class nuclei instance segmentation, $256 \times 256$ | Accurately represented, verified against raw patches. |
| **SIIM-ACR** | Chest Radiographs (Pneumothorax) | Binary classification, binary mask segmentation, $224 \times 224$ | Accurately represented. |

### Preprocessing & Normalization Findings:
- **Macenko Stain Normalization** ([`src/apply_macenko.py`](file:///home/apath/Work/temp/final/src/apply_macenko.py)):
  - `_optical_density` calculates optical density using natural logarithm $-\ln(RGB/255)$ rather than the canonical Macenko base-10 $-\log_{10}(RGB/255)$. This subtle scaling difference should be explicitly specified in the methodology.

---

## 3. Model Architecture & Loss Formulation Audit

### Architecture Implementation:
- **Backbones**: Evaluated on VGG-16 and MobileNetV2 encoders.
- **Segmentation Head**: `smp.Unet` decoder with skip connections (`--no-skip-connections` evaluated as an ablation).
- **Classification Head**: Global Average Pooling (GAP) $\to$ Linear(256) $\to$ Dropout(0.5) $\to$ Linear($K$ classes).

### Loss Formulation Discrepancies:
- **Segmentation Loss**:
  - **Manuscript Draft Claim**: $\mathcal{L}_{seg} = \mathcal{L}_{BCE} + \mathcal{L}_{Dice}$.
  - **Verified Code Ground Truth**: [`src/training.py`](file:///home/apath/Work/temp/final/src/training.py) uses `nn.BCEWithLogitsLoss` (binary) or `nn.CrossEntropyLoss` (multi-class). **No Dice loss term exists in the objective function**.
- **GradNorm Mechanism**:
  - Shared parameter set $W$ is taken as `unet.encoder.parameters()` (all shared encoder weights, not merely the final convolutional layer).
- **CLI & Parameter Overrides**:
  - Draft claims: *"the released code now exposes explicit `--enable-gradnorm` and `--disable-gradnorm` overrides"*.
  - Code truth: Only `--disable-gradnorm` exists. `apply_phase_config` in `src/main.py` overrides `use_gradnorm` based on phase.

---

## 4. Empirical Results & Statistical Matrix (26 Runs)

All 26 checkpoint summaries ([`checkpoints/summary_*.json`](file:///home/apath/Work/temp/final/checkpoints/)) were cross-referenced against `paper_results_matrix.csv` and [`paper/main.tex`](file:///home/apath/Work/temp/final/paper/main.tex):

| Run Group | Setting | Verified Metric Alignment | Key Observation |
|---|---|---|---|
| **Runs 01–06** | Single-task Baselines | Final Val Acc & Dice match CSV | Strong baseline classification without multi-task interference. |
| **Runs 07–12** | Multi-Task Uniform ($\lambda_{seg}=5.0, \lambda_{cls}=1.0$) | Final Val Acc & Dice match CSV | Noticeable Dice degradation on PANDA/PanNuke without gradient balancing. |
| **Runs 13–18** | Multi-Task + GradNorm ($\alpha=1.5$) | Final Val Acc & Dice match CSV | Run 18 comment in [`run_all_experiments.sh`](file:///home/apath/Work/temp/final/run_all_experiments.sh) line 316 contains a legacy stale comment. |
| **Runs 19–26** | Ablation Matrix (No-Macenko, No-Skip, Loss Ratios) | Final Val Acc & Dice match CSV | Validates skip-connection necessity for multi-scale feature propagation. |

### Statistical Methodology Finding:
- **Confidence Intervals**: §4.2 of `main.tex` and `paper_results_matrix_with_ci.csv` cite "Wilson 95% confidence intervals". However, [`scripts/prepare_empirical_proofs.py`](file:///home/apath/Work/temp/final/scripts/prepare_empirical_proofs.py) calculates confidence intervals using a Gaussian Normal approximation ($\hat{p} \pm 1.96 \sqrt{\hat{p}(1-\hat{p})/n}$) with hardcoded $N$ proxies. This must be aligned to rigorous Wilson score intervals before submission.

---

## 5. Immediate Action Plan for Publication Readiness

1. **Correct Manuscript Discrepancies in [`paper/main.tex`](file:///home/apath/Work/temp/final/paper/main.tex)**:
   - Fix TCGA modality description from "LGG brain MRI" to histopathology.
   - Correct PANDA segmentation description to 6-class gland architecture.
   - Align loss function equation to reflect pure BCE/CE rather than claiming unapplied Dice loss.
   - Clarify the exact Wilson vs Normal approximation formulas used for error bounds.
2. **Standardize Target Venue Formatting**:
   - The draft is currently formatted in standard LNCS (`llncs.cls`) for MICCAI. Ensure bibliography style and word count adhere to venue submission ceilings.
3. **Session Rollover in WSL Antigravity**:
   - Discontinue session `cancer_paper_audit` (poisoned by context saturation loop) and roll to `cancer_paper_audit_stage2` for subsequent LaTeX editing and proof generation.
