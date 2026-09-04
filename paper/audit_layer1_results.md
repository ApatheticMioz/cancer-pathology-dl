# Layer 1 Forensic Audit: Results Matrix & Empirical Ground Truth

## Executive Summary
This document establishes the empirical ground truth for the 26-run benchmark presented in Table 1 of the manuscript (*"All Dice, No Slice: Metric Artifacts, Data Leakage, and Task Interference in Multi-Task Computational Pathology"*). Every claim, metric, and confidence interval has been cross-verified against raw repository artifacts: `paper/paper_results_matrix.csv`, `paper/paper_results_matrix_with_ci.csv`, `checkpoints/summary_*.json`, and `checkpoints/epoch_log.jsonl`.

---

## 1. Inventory & Consistency: CSV vs. Checkpoint Artifacts
- **Consistency Score**: **100% Match** across all 26 experimental runs.
- **Verification Method**: Each row in `paper/paper_results_matrix.csv` was programmatically reconciled against its corresponding `checkpoints/summary_{idx}_*.json` artifact (`val=True`).
- **Schema**:
  - `paper_results_matrix.csv`: `Run Label`, `Phase`, `Dataset`, `Encoder`, `LR`, `Use GradNorm`, `GradNorm Alpha`, `Skip Connections`, `Macenko`, `Seg Weight`, `Cls Weight`, `Lambda Ratio (Seg:Cls)`, `Accuracy (%)`, `Macro Dice (%)`, `Paper Acc (%)`, `Paper Dice (%)`, `Acc Delta (%)`, `Dice Delta (%)`, `Status`, `Timestamp`.
  - `paper_results_matrix_with_ci.csv`: Appends `Acc 95% CI Lower`, `Acc 95% CI Upper`, `Acc 95% CI`.

All reported deltas against published baseline values (Rhanoui et al., 2025: TCGA Acc 89.0/Dice 97.0; PANDA Acc 87.0/Dice 98.0; SIIM Acc 82.0/Dice 99.0) accurately match the underlying training logs.

---

## 2. Critical Methodological Finding: Wald vs. Wilson Confidence Intervals
The manuscript repeatedly claims that reported uncertainty intervals are **"95% Wilson Score Intervals"** (e.g., `main.tex` line 246: *"95% Wilson score confidence intervals"*).

### Forensic Ground Truth
1. **The Code**: Inspection of `scripts/prepare_empirical_proofs.py` (lines 65–104, function `generate_statistical_confidence_intervals`) proves that **25 of the 26 rows compute a Normal (Wald) Asymptotic Interval**, NOT a Wilson interval:
   $$\text{CI}_{\text{Wald}} = \hat{p} \pm 1.96 \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}$$
2. **Approximate Sample Sizes**: `prepare_empirical_proofs.py` hardcodes approximate validation sizes:
   - `TCGA`: 786 (Actual validation split in JSON: **778**)
   - `PANDA`: 2104 (Actual validation split in JSON: **2104**)
   - `SIIM`: 2409 (Actual validation split in JSON: **2135**)
   - `PANNUKE`: 1500 (Actual validation split in JSON: **1567**)
3. **The Single Exception**: Only **Run 18** (the PANDA GradNorm rerun from Sept 3, `scripts/run_canonical_gradnorm_panda.py` lines 41–49) computed a true Wilson score interval.
4. **Impact on Manuscript Claims**:
   - **Macenko "Disjoint CIs" claim** (`main.tex` line 240): The non-overlapping CIs ([32.67, 36.73] vs. [38.11, 42.31] on PANDA; [95.77, 97.59] vs. [98.96, 99.76] on PanNuke) are Wald intervals computed with approximate $n$.
   - **Skip-Connection Null claim** (`main.tex` lines 234–237): The overlapping intervals ([91.57, 95.07] vs. [92.72, 95.96] on TCGA; [32.67, 36.73] vs. [33.27, 37.35] on PANDA) are likewise Wald intervals.
   - **Action Item for Revision**: The manuscript text must be corrected to state that intervals were computed via normal approximation with validation split sample sizes, or recomputed rigorously using true Wilson score intervals with exact $n$.

---

## 3. The SIIM-ACR Pneumothorax Metric Artifact Proof
All 4 SIIM-ACR benchmark runs (Runs 05, 06, 11, 12) report the exact byte-identical float:
$$\text{best\_val\_dice} = 0.7774375410222295 \quad (77.74\%)$$

### Mechanism Proven in Code & Logs
1. **Empty-Mask Credit Convention (`src/metrics.py` lines 25–48)**:
   When both predicted segmentation mask and ground-truth mask are empty ($\text{union} == 0$), the metric implementation assigns:
   $$\text{empty\_score} = 1.0$$
2. **Total Model Segmentation Failure (`checkpoints/epoch_log.jsonl`)**:
   Across all 50 epochs of training on SIIM, the multi-task UNet predicted all zeros for segmentation. From Epoch 1 to Epoch 50, `epoch_log.jsonl` shows `vl_dice = 0.777438` at every single epoch.
3. **The 77.74% Value**:
   In the SIIM validation split ($N = 2,135$), exactly **1,659 slices (77.70%)** contain zero pneumothorax lesions ($\rho = 1659 / 2135 = 0.777049...$).
   The unweighted mean of batch-level Dice scores across 67 validation batches ($66 \times 32 + 1 \times 23$) evaluates to exactly `0.7774375410222295`.
4. **Paper Verification**:
   The manuscript's core thesis—that reported 77.7% validation Dice is a pure empty-mask artifact rather than genuine lesion segmentation—is **empirically 100% TRUE**.

---

## 4. Adjudication of Specific Manuscript Claims

| Manuscript Claim | Claimed Values | Repository Ground Truth | Status | Exact Provenance |
| :--- | :--- | :--- | :--- | :--- |
| **Run 18 Collapse** | Acc: 45.39% $\to$ 29.04%<br>Dice: 44.08% $\to$ 31.43% | Acc: 45.39% $\to$ 29.04%<br>Dice: 44.08% $\to$ 31.43% | **TRUE** | `paper_results_matrix.csv` (Row 18 & 20); `summary_18_g4_panda_isolate_gn_true.json` |
| **Skip Ablation (TCGA)** | Acc: 93.32% $\to$ 94.34%<br>Dice: 84.20% $\to$ 84.12% | Acc: 93.32% $\to$ 94.34%<br>Dice: 84.20% $\to$ 84.12% | **TRUE** | `paper_results_matrix.csv` (Row 8 & 25) |
| **Skip Ablation (PANDA)** | Acc: 34.70% $\to$ 35.31%<br>Dice: 31.34% $\to$ 28.94% | Acc: 34.70% $\to$ 35.31%<br>Dice: 31.34% $\to$ 28.94% | **TRUE** | `paper_results_matrix.csv` (Row 10 & 26) |
| **Macenko Removal Delta** | PANDA: +5.51 points<br>PanNuke: +2.68 points | PANDA: 34.70% $\to$ 40.21% (+5.51%)<br>PanNuke: 96.68% $\to$ 99.36% (+2.68%) | **TRUE** | `paper_results_matrix.csv` (Row 10 vs 23; Row 16 vs 24) |
| **PANDA Empirical Range** | 29.04% – 46.15% | Min: 29.04% (Run 18)<br>Max: 46.15% (Run 4) | **TRUE** | `paper_results_matrix.csv` |
| **Wilson CI Attribution** | "95% Wilson CIs" | 25/26 rows are Wald intervals with approximate $n$ | **METHOD ERROR IN PAPER** | `scripts/prepare_empirical_proofs.py:76-104` |
