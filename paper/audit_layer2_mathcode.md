# Layer 2 Forensic Audit: Math, Code AST, and Architectural Provenance

## Executive Summary
This document provides a line-by-line AST code audit verifying the mathematical formulations, loss balancing mechanisms, network architectures, and evaluation metrics implemented across the computational pathology codebase (`/home/apath/Work/temp/final`). Every finding is cross-referenced with exact file and line provenance.

---

## 1. Training Objective & Loss Formulations
- **Exact File & Line**: `src/training.py` lines 125–148, `src/models.py` lines 20–55.
- **Formulation**: The multi-task network is trained on a joint weighted sum of classification and segmentation losses:
  $$\mathcal{L}_{\text{total}} = \lambda_{\text{seg}} \mathcal{L}_{\text{seg}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}$$
  Where:
  - Classification loss $\mathcal{L}_{\text{cls}}$ is standard Multi-Class Cross-Entropy (`nn.CrossEntropyLoss`).
  - Segmentation loss $\mathcal{L}_{\text{seg}}$ is Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`) for single-class tasks, or multi-class Cross-Entropy for multi-organ segmentation.
- **Soft-Dice in Training**: **NONE.** There is **zero** soft-Dice or Dice-loss term in the backward pass during training. Dice is computed exclusively as a post-hoc evaluation metric in validation loops (`src/metrics.py`).
- **Lambda Parameterization & Ratios**:
  - The benchmark supports ratios: `1:1`, `5:1`, `1:10`, `10:1` via CLI flags `--lambda_seg` and `--lambda_cls`.
  - Default benchmark setting across the 26 runs is $\lambda_{\text{seg}} = 5.0, \lambda_{\text{cls}} = 1.0$ (ratio 5:1).

---

## 2. Network Architecture & Task Split
- **Exact File & Line**: `src/models.py` lines 58–110 (`class MultiTaskUNet`).
- **Shared Encoder Backbones**:
  - Integrated via `segmentation_models_pytorch.Unet` with ImageNet-pretrained weights (`encoder_weights="imagenet"`).
  - Backbones supported: `vgg16` and `mobilenet_v2`.
- **Task Split Point**:
  - Features are extracted by the shared encoder: `features = self.unet.encoder(x)`.
  - The task split occurs at the final bottleneck layer:
    ```python
    bottleneck = features[-1]
    cls_out = self.cls_head(bottleneck)
    decoder_out = self.unet.decoder(features)
    seg_out = self.unet.segmentation_head(decoder_out)
    ```
- **Classification Head**: Global Average Pooling followed by a 2-layer MLP:
  ```python
  self.cls_head = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten(),
      nn.Linear(bottleneck_dim, 256),
      nn.ReLU(inplace=True),
      nn.Dropout(0.5),
      nn.Linear(256, num_classes),
  )
  ```
- **Skip-Connection Ablation Mechanism (`src/models.py` lines 101–105)**:
  When `--no_skip_connections` is toggled (`self._skip_connections = False`), intermediate encoder feature maps passed to the UNet decoder are replaced with tensors of zeros:
  ```python
  if not self._skip_connections:
      features = [torch.zeros_like(f) for f in features[:-1]] + [bottleneck]
  ```

---

## 3. Dice Evaluation & The SIIM Invariant Floor Mechanics
- **Exact File & Line**: `src/metrics.py` lines 18–50 (`dice_coefficient`), `src/training.py` line 193.
- **Empty-Mask Convention**:
  When both predicted mask and ground truth mask contain zero foreground pixels ($\text{union} == 0$), the implementation explicitly assigns:
  $$\text{empty\_score} = 1.0$$
- **Averaging Order**:
  `dice_coefficient` computes the mean score per batch. `src/training.py` records these batch scores in `dice_vals` and takes an unweighted average of batch means: `float(np.mean(dice_vals))`.
- **Mechanistic Proof of Byte-Identical SIIM Dice (`0.7774375410222295`)**:
  1. Across all 4 SIIM runs (Runs 05, 06, 11, 12), the segmentation head experienced complete gradient starvation / negative-class collapse, predicting all zeros across all 50 epochs (`checkpoints/epoch_log.jsonl`).
  2. For empty ground-truth slices, $\text{pred} = 0, \text{target} = 0 \implies \text{Dice} = 1.0$.
  3. For positive ground-truth slices, $\text{pred} = 0, \text{target} > 0 \implies \text{Dice} = 0.0$.
  4. The SIIM validation set contains $N = 2,135$ slices, of which exactly **1,659 slices (77.70%)** are empty.
  5. With batch size 32, the validation set is partitioned into 66 batches of 32 slices and 1 final batch of 23 slices (67 batches total). The unweighted average of batch means under this exact slice distribution evaluates to the float `0.7774375410222295`.
  6. **Conclusion**: The reported 77.74% Dice is an uninformative metric artifact reflecting dataset background sparsity ($\\rho \\approx 0.78$), not lesion detection capability.

---

## 4. GradNorm Implementations: Benchmark vs. Canonical

### A. The 26-Run Benchmark Variant (`src/models.py` & `src/training.py`)
- **Weight Parameterization**: Log space (`self.log_weights = nn.Parameter(torch.log(init))`), normalized via `normalize_()` to sum to 2.0.
- **Optimization Strategy**: **Joint Single-Optimizer.** Task weights are updated by the **same primary Adam optimizer** driving the network parameters:
  ```python
  clip_params = list(model.parameters())
  if gradnorm is not None and not static_weights:
      clip_params += list(gradnorm.parameters())
  torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
  scaler.step(optimizer)
  ```
- **Asymmetry Parameter**: Fixed at $\alpha = 1.5$.
- **Disparity Loss**: $\mathcal{L}_{\text{grad}} = \sum_i |w_i G_W^{(i)} - \bar{G}_W \times [r_i]^\alpha|$.

### B. The Standalone Canonical Variant (`scripts/run_canonical_gradnorm_panda.py`)
- **Optimization Strategy**: **Decoupled Dual-Optimizer.**
  - Network optimizer: `optimizer_model = Adam(model.parameters(), lr=1e-3)`.
  - Task weight optimizer: `optimizer_weights = Adam([weights], lr=0.025)`.
- **Strict Gradient Detachment**:
  Network parameters are explicitly detached when calculating gradient norms (`norms = torch.stack([norm_seg, norm_cls]).detach()`), preventing $\mathcal{L}_{\text{grad}}$ from contaminating network parameter gradients.
- **Empirical Results Recorded**: Acc $\approx 43.06\%$, Dice $\approx 39.98\%$ (logged in `logs/canonical_gradnorm_run18.log`).

---

## 5. Macenko Stain Normalization
- **Exact File & Line**: `src/apply_macenko.py` lines 50–95.
- **Optical Density Formulation**:
  Implemented using the **natural logarithm**:
  ```python
  def _optical_density(rgb: np.ndarray) -> np.ndarray:
      img = np.clip(rgb.astype(np.float32), 1.0, 255.0)
      return -np.log(img / 255.0)
  ```
- **Stain Vector Extraction**:
  Covariance matrix $\mathbf{C} = \text{Cov}(\text{OD})$, eigen-decomposition via `np.linalg.eigh(cov)` to find the two largest eigenvectors, projection of OD data onto the plane, angular calculation via `np.arctan2`, robust boundary clipping at 1st and 99th percentiles (`np.percentile(angles, [1.0, 99.0])`), and unit-vector normalization.

---

## 6. Dataset Protocol & Patient-Level Leakage Safeguards
- **Exact File & Line**: `src/data.py` lines 40–120.
- **Group-Aware Splitting**:
  Uses `GroupShuffleSplit` / `GroupKFold` grouped strictly on `patient_id` or `slide_id` across TCGA and PANDA datasets to guarantee zero cross-contamination of tiles from the same patient between training and validation splits.
- **Image Resolution**: Resized to $256 \times 256$ pixels across all benchmarks.
