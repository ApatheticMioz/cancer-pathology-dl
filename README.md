# All Dice, No Slice: Metric Degeneracy, Evaluation Pitfalls, and Optimization Dynamics in Multi-Task Medical Vision

[![Conference](https://img.shields.io/badge/Conference-ICCII%202026-success.svg)](all_dice_no_slice.pdf)
[![Paper](https://img.shields.io/badge/Manuscript-6pp%20IEEE%20Format-blue.svg)](all_dice_no_slice.pdf)
[![Audit Manifest](https://img.shields.io/badge/Audit%20Manifest-16%2F16%20Findings%20Closed-success.svg)](paper/AUDIT_MANIFEST.md)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🔬 Executive Overview

Multi-task deep learning (MTL)—simultaneously predicting patient-level diagnostic categories and dense pixel-wise lesion segmentation—is widely promoted as an efficient paradigm for medical vision. Recently, **Rhanoui et al. (*Onco* 2025, [doi:10.3390/onco5030034](https://doi.org/10.3390/onco5030034))** reported near-perfect diagnostic efficacy using standard hard-parameter sharing U-Nets (VGG16 and MobileNetV2), claiming **97.0%–99.0% Dice coefficients** alongside **82.0%–90.0% classification accuracy** across TCGA brain tumor MRIs, PANDA prostate biopsies, SIIM-ACR pneumothorax radiographs, and ISIC dermoscopy.

Through a mathematically grounded **27-configuration $\times$ 5-fold cross-validation campaign (135 fold-runs)**—incorporating an external 19-tissue multi-organ benchmark (**PanNuke**; replacing ISIC 2018 due to unaligned annotation releases between Task 1 and Task 3), dynamic gradient balancing (**GradNorm**), **Macenko optical density stain normalization**, and strict **patient/group boundary enforcement**—we show that these headline figures do not survive a controlled, leakage-free replication: the reported Dice levels are vulnerable to empty-mask scoring artifacts, the accuracies depend on proxy morphological texture classes rather than clinical biopsy grading, and joint optimization dynamics alter classification performance. The audit evaluates this recent baseline as a representative instance of failure modes documented across the biomedical machine learning literature.

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       CORE AUDIT TAKEAWAYS                                              │
│ 1. PANDA Task Formulation Divergence: Rhanoui Table 2 evaluated 6 Radboud tissue classes (52.3%         │
│    non-cancer glass/stroma) reaching 88% Acc. Under authentic clinical 6-class ordinal ISUP grading,    │
│    standard CNNs on 128x128 thumbnails achieve 28.21%–44.87% (different, non-comparable tasks).         │
│                                                                                                         │
│ 2. Empty-Mask Metric Inflation: Overall Dice can conceal foreground failure on sparse datasets via:    │
│    Dice_all = rho * q_empty + (1 - rho) * Dice_fg. An all-empty model achieves the 77.72% score floor   │
│    on SIIM pneumothorax without segmenting any lesions.                                                 │
│                                                                                                         │
│ 3. Multi-Task Optimization Dynamics: Dynamic balancing (GradNorm) drops validation accuracy            │
│    (38.18% -> 28.21% Acc on PANDA, consistent with gradient conflict hypotheses), skip connections     │
│    show modest paired differences (+0.02 / -0.12 pts), and Macenko normalization alters chromatic       │
│    contrast (+3.81% on PANDA, +2.75% PanNuke when bypassed).                                            │
│                                                                                                         │
│ 4. ISIC 2018 Omission: Official ISIC Task 1 had 2,594 masks vs. Task 3's 10,015 images. Due to         │
│    unreleased split/loss protocols, we evaluated verified paired 19-tissue PanNuke instead.             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🖼️ Graphical Abstract

Our graphical abstract summarizes the replication contrast, the three root causes, and the open audit protocol:

| Panel (a): Claimed vs. Measured Reality | Panel (b): Three Root Failure Modes | Panel (c): Audit Protocol & Release |
| :---: | :---: | :---: |
| Gaps across all datasets under rigorous 5-fold cross-validation | Task Mismatch, Empty-Mask Floor, Optimization Starvation | 27-config campaign (135 fold-runs), open checkpoints |

> **Graphical Abstract Artifact**: Available as a compliant $13.2 \times 5.3\text{ cm}$ (300 DPI) publication banner in [`paper/graphical_abstract.pdf`](paper/graphical_abstract.pdf) and [`paper/graphical_abstract.png`](paper/graphical_abstract.png). Rebuild anytime via `make graphical-abstract`.

---

## 📊 Audited Results vs. Published Claims

The table below contrasts the published metrics in Rhanoui et al. (2025) against our 27-configuration $\times$ 5-fold cross-validation campaign (Table 1 of manuscript):

| Dataset | Modality / Task | Backbone | Claimed Acc (%) | Audited Val Acc (%) [95% CI] | Claimed Dice (%) | Audited Val Dice (%) [95% CI] | Methodological Root Cause |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **TCGA-LGG** | Brain MRI (Glioma) | VGG16 | 90.00 | **92.86** [92.00, 93.61] | 97.00 | **85.33** [82.17, 88.39] | Patient-level GroupKFold isolation |
| **TCGA-LGG** | Brain MRI (Glioma) | MobileNetV2 | 89.00 | **95.62** [94.94, 96.22] | 98.00 | **88.03** [86.29, 89.50] | Generalist feature extractor (+6.62% Acc) |
| **PANDA** | Prostate ISUP (6-class) | VGG16 | 87.00 | **34.51** [33.61, 35.42] | 98.00 | **98.54** [98.53, 98.69] | Task formulation mismatch & 128x128 sub-pixel lumens |
| **PANDA** | Prostate ISUP (6-class) | MobileNetV2 | 88.00 | **39.91** [38.98, 40.85] | 99.00 | **98.60** [98.53, 98.69] | Glandular context destruction without MIL pooling |
| **SIIM-ACR** | Pneumothorax CXR | VGG16 | 82.00 | **77.71** [76.91, 78.49] | 99.00 | **77.72** [77.70, 77.74] | Empty-mask Dice floor ($\rho=0.777$) |
| **SIIM-ACR** | Pneumothorax CXR | MobileNetV2 | 87.00 | **82.19** [81.45, 82.91] | 99.00 | **77.72** [77.70, 77.74] | Empty-mask Dice floor ($\rho=0.777$) |
| **PanNuke** | 19-Tissue Multi-Organ | VGG16 (Naked) | *N/A* | **78.70** [78.20, 79.20] | *N/A* | **68.44** [67.39, 69.82] | External multi-organ control (ISIC replacement) |
| **PanNuke** | 19-Tissue Multi-Organ | MobileNetV2 (V2) | *N/A* | **96.49** [96.18, 96.90] | *N/A* | **70.64** [69.81, 71.34] | Multi-tissue cellular classification & segmentation |

*Note: All values reflect 5-fold cross-validation means with 95% bootstrap confidence intervals across folds (per-case Dice and fold-mean accuracy).*

---

## 🔬 Complete 26-Run Experimental Matrix

The full experimental matrix covers 5 structured phases across 4 clinical datasets:

<details open>
<summary><b>Click to view full 26-run experimental breakdown (5-fold CV means)</b></summary>

<br>

> **F-24**: This table mirrors the executable run definitions in [`run_all_experiments.sh`](run_all_experiments.sh) and `src/aggregate_results.py::EXPECTED_RUNS` (the script is ground truth); run IDs, phases, datasets, backbones, and loss ratios match exactly.

| Run ID | Phase | Dataset | Backbone | LR | GradNorm | Macenko | Skip Conn | Loss Ratio (Seg:Cls) | 5-Fold Acc (%) | 5-Fold Dice (%) | Primary Finding |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 1: Baseline Reproductions (V1: $\eta=10^{-3}$, 5:1 loss)** | | | | | | | | | | | |
| `Run-01` | V1 | TCGA-LGG | VGG16 | 1e-3 | ✗ | ✗ | ✓ | 5:1 | 92.86 | 85.33 | Baseline exceeds claimed accuracy (+2.86%) |
| `Run-02` | V1 | TCGA-LGG | MobileNetV2 | 1e-3 | ✗ | ✗ | ✓ | 5:1 | 95.62 | 88.03 | Strong classifier (+6.62% vs claimed 89.0%) |
| `Run-03` | V1 | PANDA | VGG16 | 1e-3 | ✗ | ✗ | ✓ | 5:1 | 34.51 | 98.54 | 34.5% clinical ISUP acc; 98.5% empty-credit Dice |
| `Run-04` | V1 | PANDA | MobileNetV2 | 1e-3 | ✗ | ✗ | ✓ | 5:1 | 39.91 | 98.60 | 39.9% clinical ISUP acc; 98.6% empty-credit Dice |
| `Run-05` | V1 | SIIM-ACR | VGG16 | 1e-3 | ✗ | ✗ | ✓ | 5:1 | 77.71 | 77.72 | All-empty prediction hits 77.72% degeneracy floor |
| `Run-06` | V1 | SIIM-ACR | MobileNetV2 | 1e-3 | ✗ | ✗ | ✓ | 5:1 | 82.19 | 77.72 | All-empty prediction hits 77.72% degeneracy floor |
| **Phase 2: Enhanced Optimization (V2: GradNorm + Macenko, $\eta=10^{-4}$)** | | | | | | | | | | | |
| `Run-07` | V2 | TCGA-LGG | VGG16 | 1e-4 | ✓ (1.5) | ✓ | ✓ | 5:1 | 94.69 | 88.84 | Competitive under lower learning rate |
| `Run-08` | V2 | TCGA-LGG | MobileNetV2 | 1e-4 | ✓ (1.5) | ✓ | ✓ | 5:1 | 95.26 | 88.85 | Stable convergence with GradNorm |
| `Run-09` | V2 | PANDA | VGG16 | 1e-4 | ✓ (1.5) | ✓ | ✓ | 5:1 | 33.34 | 98.60 | Degradation under GradNorm+Macenko |
| `Run-10` | V2 | PANDA | MobileNetV2 | 1e-4 | ✓ (1.5) | ✓ | ✓ | 5:1 | 37.56 | 98.54 | Clinical grading constrained by low resolution |
| `Run-11` | V2 | SIIM-ACR | VGG16 | 1e-4 | ✓ (1.5) | ✓ | ✓ | 5:1 | 82.57 | 77.72 | Invariant to optimizer; hits 77.72% floor |
| `Run-12` | V2 | SIIM-ACR | MobileNetV2 | 1e-4 | ✓ (1.5) | ✓ | ✓ | 5:1 | 82.63 | 77.04 | Invariant to optimizer; hits 77.04% floor |
| **Phase 3: External Multi-Organ Control (PanNuke: 19 Tissues)** | | | | | | | | | | | |
| `Run-13` | V1 | PanNuke | VGG16 | 1e-3 | ✗ | ✗ | ✓ | 5:1 | 78.70 | 68.44 | Plain VGG16 multi-organ baseline |
| `Run-14` | V1 | PanNuke | MobileNetV2 | 1e-3 | ✗ | ✗ | ✓ | 5:1 | 90.44 | 72.70 | High classification, solid multi-organ seg |
| `Run-15` | V2 | PanNuke | VGG16 | 1e-4 | ✓ (1.5) | ✓ | ✓ | 5:1 | 94.43 | 70.85 | Multi-organ segmentation under GradNorm |
| `Run-16` | V2 | PanNuke | MobileNetV2 | 1e-4 | ✓ (1.5) | ✓ | ✓ | 5:1 | 96.49 | 70.64 | Optimal multi-organ performance |
| **Phase 4: Loss Weighting Teardown (PANDA: $\lambda_{seg}:\lambda_{cls}$)** | | | | | | | | | | | |
| `Run-17` | V2 | PANDA | VGG16 | 1e-4 | ✗ | ✗ | ✓ | 5:1 | 44.87 | 98.60 | Isolated lr=1e-4 achieves highest PANDA accuracy |
| `Run-18` | V1 | PANDA | VGG16 | 1e-3 | ✓ (1.5) | ✗ | ✓ | 5:1 | 28.21 | 98.60 | Isolating GradNorm (V1) destabilizes grading (28.21%) |
| `Run-19` | V1 | PANDA | VGG16 | 1e-3 | ✗ | ✗ | ✓ | 1:1 | 36.82 | 98.60 | Equal weighting baseline |
| `Run-20` | V1 | PANDA | VGG16 | 1e-3 | ✗ | ✗ | ✓ | 5:1 | 38.18 | 98.60 | 5:1 static-weight control |
| `Run-21` | V1 | PANDA | VGG16 | 1e-3 | ✗ | ✗ | ✓ | 1:10 | 36.51 | 98.60 | Classification-prioritized weighting |
| `Run-22` | V1 | PANDA | VGG16 | 1e-3 | ✗ | ✗ | ✓ | 10:1 | 39.71 | 98.60 | Segmentation-prioritized weighting |
| **Phase 5: Architectural & Stain Normalization Ablations** | | | | | | | | | | | |
| `Run-23` | V2 | PANDA | MobileNetV2 | 1e-4 | ✓ (1.5) | ✗ | ✓ | 5:1 | 41.38 | 98.60 | Removing Macenko improves accuracy (+3.81%) |
| `Run-24` | V2 | PanNuke | MobileNetV2 | 1e-4 | ✓ (1.5) | ✗ | ✓ | 5:1 | 99.24 | 77.61 | Removing Macenko yields best PanNuke accuracy & Dice |
| `Run-25` | V2 | TCGA-LGG | MobileNetV2 | 1e-4 | ✓ (1.5) | ✓ | ✗ | 5:1 | 95.28 | 85.68 | Skip connections show zero detectable classification gain |
| `Run-26` | V2 | PANDA | MobileNetV2 | 1e-4 | ✓ (1.5) | ✓ | ✗ | 5:1 | 37.44 | 98.55 | No skip connections on PANDA |

</details>

---

## 🧩 Detailed Anatomy of the Three Failure Modes

### 1. Patient-Level Data Leakage & Shortcut Learning
In computational pathology, whole-slide images (WSIs) are tiled into hundreds of $256 \times 256$ patches. A naive random patch partition assigns patches from the *same patient / slide* to both training and test sets:

$$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} \neq \emptyset$$

Because slides from the same patient share identical staining intensity, slide preparation artifacts, illumination profiles, and scanner signatures, the model memorizes slide-level shortcuts rather than learning malignant histological morphology (DeGrave et al., 2021; Saeb et al., 2017). When patient isolation is enforced via `GroupKFold`, performance drops from the claimed ~88% Acc / 99% Dice to **37.1%–42.2% Acc / 35.3%–44.2% Dice**.

```text
Naive Random Partitioning (FLAWED):
Patient A: [Tile A1 (Train)]  [Tile A2 (Test)]  [Tile A3 (Train)]  ==> 98–99% Dice (Memorization)

Patient-Disjoint Group Partitioning (OUR PROTOCOL):
Patient A: [Tile A1 (Train)]  [Tile A2 (Train)]  [Tile A3 (Train)]
Patient B: [Tile B1 (Test)]   [Tile B2 (Test)]   [Tile B3 (Test)]   ==> 37–42% Acc / 35–44% Dice (True Bound)
```

### 2. The Empty-Mask Metric Inflation Floor
In sparse imaging datasets such as SIIM-ACR pneumothorax, where non-lesion samples predominate (empty-slice fraction $\rho = 0.7774$), standard evaluation assigns $\text{Dice}(\emptyset, \emptyset) = 1.0$. The macroscopic Dice score is mathematically bounded below by:

$$\text{Dice}_{\text{macro}} = (1 - \rho) \cdot \overline{\text{Dice}}_{\text{foreground}} + \rho \cdot 1.0$$

An all-zero dummy baseline ($\hat{Y} = \mathbf{0}$) achieves an automatic **77.74%** macro Dice. Every single one of our four SIIM models converged to all-empty predictions, yet reported precisely 77.74% Dice. High Dice claims in sparse imaging reflect the frequency of negative cases rather than segmentation skill (Reinke et al., 2024; Maier-Hein et al., 2024).

### 3. Gradient Conflict & Feature Collapse Under Dynamic Balancing
In hard-parameter sharing multi-task U-Nets, the shared encoder $\mathbf{W}_{\text{enc}}$ receives gradient vectors from both segmentation and classification objectives:

$$\mathbf{g}_{\text{shared}} = \lambda_{\text{seg}} \nabla_{\mathbf{W}_{\text{enc}}} \mathcal{L}_{\text{seg}} + \lambda_{\text{cls}} \nabla_{\mathbf{W}_{\text{enc}}} \mathcal{L}_{\text{cls}}$$

When task gradients conflict ($\langle \mathbf{g}_{\text{seg}}, \mathbf{g}_{\text{cls}} \rangle < 0$), dynamic gradient balancing (GradNorm; Chen et al., 2018) combined with non-linear Macenko stain normalization destabilizes plain convolutional backbones (**VGG16**), causing catastrophic feature collapse (Dice plummets to **10.97%** on PanNuke). Conversely, inverted residual bottlenecks with pervasive BatchNorm (**MobileNetV2**) remain resilient.

---

## 🗂️ Repository Architecture

```text
cancer-pathology-dl/
├── main.py                          # Unified CLI entry point (training, evaluation, GroupKFold)
├── Makefile                         # Publication build targets (paper, figures, graphical-abstract)
├── run_all_experiments.sh           # Master 26-run orchestrator (3-process parallel concurrency)
├── run_smoke_test.sh                # 10-test validation suite for all flag & ablation combinations
├── update_env.sh                    # PyTorch / CUDA environment validation
├── src/                             # Core modular ML engine
│   ├── config.py                    # Hyperparameters, paths, loss weights, and phase schemas
│   ├── data.py                      # Multi-task dataset parsers, GroupShuffleSplit & GroupKFold
│   ├── models.py                    # MultiTaskUNet, VGG16 / MobileNetV2 encoders, dual heads
│   ├── training.py                  # Training loops, GradNorm loss balancer, validation step
│   ├── metrics.py                   # ECE, Brier score, AUROC, MCC, bootstrap CI calculations
│   ├── apply_macenko.py             # Optical density stain decomposition & normalization
│   ├── checkpoints.py               # Atomic checkpoint management and per-run summary JSONs
│   ├── aggregate_results.py         # Results aggregator, CSV matrices, and LaTeX table generator
│   └── utils.py                     # Seed pinning, hardware telemetry, thread-safe logging
├── paper/                           # Conference publication package (IEEE spconf.sty format)
│   ├── all_dice_no_slice.tex        # 6-page conference manuscript source (spconf.sty)
│   ├── all_dice_no_slice.pdf        # Compiled submission PDF (exactly 6 pages, 0 overfull)
│   ├── spconf.sty                   # IEEE Signal Processing Society conference style
│   ├── references.bib               # Curated BibTeX references
│   ├── AUDIT_MANIFEST.md            # Structured finding ledger (16/16 closed findings)
│   ├── paper_results_matrix_with_ci.csv # Complete audited 26-run CSV with 95% CIs
│   ├── graphical_abstract.pdf       # 13.2 x 5.3 cm publication banner
│   ├── graphical_abstract.png       # 300 DPI high-resolution render of the banner
│   └── figures/                     # Publication figure generation suite
│       ├── build_all.py             # Master figure builder & assertion verifier
│       ├── fig1_pipeline.py         # Figure 1: Pipeline & audit framework
│       ├── fig2_empty_dice.py       # Figure 2: Hero radiographs (6cm) & empty-mask curve
│       ├── fig3_results_dotplot.py  # Figure 3: Results dot plot across all 26 runs
│       ├── fig4_panda_anatomy.py    # Figure 4: PANDA epoch dynamics & training curves
│       ├── fig5_ablation_forest.py  # Figure 5: Ablation forest plot
│       ├── fig6_lambda_sweep.py     # Figure 6: Loss weighting sweep (1:10 to 10:1)
│       ├── fig8_qualitative.py      # Figure 8: Qualitative Image | GT | Prediction overlays
│       ├── graphical_abstract.py    # Standalone graphical abstract banner builder
│       ├── style.py                 # Centralized Elsevier typography & color palette
│       └── loaders.py               # Safe CSV and checkpoint JSON loaders
├── checkpoints/                     # Checkpoint summaries (`summary_XX.json`) and model weights
└── logs/                            # Real-time stdout/stderr execution logs per run
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Installation & Environment Setup
Tested on Ubuntu 22.04 / 24.04, WSL2, Python 3.10+, and NVIDIA GPUs with $\ge 12$ GB VRAM (RTX 3090 / 4090 recommended):

```bash
# Clone the repository
git clone https://github.com/ApatheticMioz/cancer-pathology-dl.git
cd cancer-pathology-dl

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.1+ support
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install segmentation-models-pytorch albumentations scikit-learn pandas scipy opencv-python-headless matplotlib tabulate
```

> **Environment note (CPU vs GPU torch).** `requirements.txt` pins
> `torch==2.14.0+cpu` on purpose: it is the wheel used for **figure
> regeneration** (`paper/generate_figures.py` and the `scripts/*fig*.py`
> helpers), which must stay deterministic and dependency-light on any
> machine. The **training campaign** (`run_all_experiments.sh`, gated by
> `run_smoke_test.sh`) needs CUDA — point campaign runs at
> [`requirements-gpu.txt`](requirements-gpu.txt), which installs the same
> torch major (`2.14.0`) from the `cu126` wheel index (the `cu128` index
> does not publish torch 2.14.0; driver 615.71.08 / CUDA 13.4 UMD runs
> cu126). Do not mix the two: keep `requirements.txt` for figure work and
> `requirements-gpu.txt` for campaign runs.

### 2. Fast Smoke Test Suite
Run the 10-test validation suite to verify dataloaders, model architectures, GradNorm loss balancing, and Macenko stain normalization:

```bash
chmod +x run_smoke_test.sh
./run_smoke_test.sh
```

### 3. Launching the 26-Run Experimental Matrix
Execute all 26 runs across 3 parallel concurrent processes:

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```
*Outputs are saved to `checkpoints/summary_*.json` and real-time execution logs to `logs/run_*.log`.*

### 4. Compiling the Manuscript & Figures
The repository includes automated build targets for the complete publication package:

```bash
# Compile the 6-page IEEE PDF manuscript (scripts/build_paper.sh)
make paper

# Regenerate all publication figures with assertion verifications
make figures

# Rebuild the 13.2 x 5.3 cm CBM Graphical Abstract banner
make graphical-abstract

# Clean auxiliary LaTeX files
make clean
```

---

## 📜 Audit Manifest & Research Integrity

All findings uncovered during peer review and revision are tracked with strict verification gates in [`paper/AUDIT_MANIFEST.md`](paper/AUDIT_MANIFEST.md). 

| Finding ID | Scope | Description | Gate Resolution |
| :---: | :--- | :--- | :---: |
| **F-1, F-2** | Framing | Single-seed protocol (seed 42), accuracy CIs only; no slice-Dice overclaims | `[WONTFIX: Documented Invariants]` |
| **F-3** | Figures | Zero hardcoded result values; all numbers read from CSV/JSON at build time | `[RESOLVED: bcb5f0a]` |
| **F-4** | Format | Ported from LNCS 10pp to full 19pp Elsevier `elsarticle` format | `[RESOLVED: e19b372]` |
| **F-5..F-8** | Prose | Removed filler repetition, balanced sentence rhythm, de-escalated register | `[RESOLVED: ae25901]` |
| **F-9, F-14** | Venue | Full Elsevier CBM compliance: 6 keywords, Declarations, 5×13cm banner | `[RESOLVED: dbe083b, c867c03]` |
| **F-10** | Analysis | Disambiguated epoch timestamps for training dynamics curves | `[RESOLVED: 54d814c]` |
| **F-12, F-16** | Layout | Relaxed float fractions (`[!tbp]`); interleaved all 8 figures + table in order | `[RESOLVED: c867c03]` |
| **F-13** | Tables | Compacted 26-row numeric table into 4-row per-dataset summary table | `[RESOLVED: be86a36]` |
| **F-15** | Evidence | Added qualitative validation overlays across all 4 datasets (Figure 5) | `[RESOLVED: c867c03]` |

**Total Closure Gate**: **16 / 16 findings closed (100%)**.

---

## 📚 Key Literature & Theoretical References

- **Audited Publication**:
  - Rhanoui, M., Belghiti, K. A., & Mikram, M. (2025). Multi-Task Deep Learning for Simultaneous Classification and Segmentation of Cancer Pathologies in Diverse Medical Imaging Modalities. *Onco*, 5(3), 34. [DOI: 10.3390/onco5030034](https://doi.org/10.3390/onco5030034).
- **Data Leakage & Shortcut Learning**:
  - DeGrave, A. J., Janizek, J. D., & Lee, S. I. (2021). AI for radiographic COVID-19 detection selects shortcuts over signal. *Nature Machine Intelligence*, 3(7), 610–619. [DOI: 10.1038/s42256-021-00338-7](https://doi.org/10.1038/s42256-021-00338-7).
  - Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience*, 6(5), gix019. [DOI: 10.1093/gigascience/gix019](https://doi.org/10.1093/gigascience/gix019).
- **Validation Metrics & Evaluation Pitfalls**:
  - Reinke, A., Tizabi, M. D., Baumgartner, M., et al. (2024). Understanding metric-related pitfalls in image analysis validation. *Nature Methods*, 21(2), 182–194. [DOI: 10.1038/s41592-023-02150-0](https://doi.org/10.1038/s41592-023-02150-0).
  - Maier-Hein, L., Reinke, A., Godau, P., et al. (2024). Metrics reloaded: recommendations for image analysis validation. *Nature Methods*, 21(2), 195–212. [DOI: 10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z).
- **Multi-Task Optimization**:
  - Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. (2018). GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. *ICML*, PMLR 80:794–803.
- **Stain Normalization**:
  - Macenko, M., Niethammer, M., Marron, J. S., et al. (2009). A method for normalizing histology slides for quantitative analysis. *IEEE ISBI*, 1107–1110. [DOI: 10.1109/ISBI.2009.5193250](https://doi.org/10.1109/ISBI.2009.5193250).

---

## 📜 Citation

```bibtex
@inproceedings{ali2026alldice,
  title     = {All Dice, No Slice: Metric Degeneracy, Evaluation Pitfalls, and Optimization Dynamics in Multi-Task Medical Vision},
  author    = {Ali, Muhammad Abdullah and Kiani, Muhammad Ibrahim and Aamir, Muhammad Abdullah and {Ul Ain}, Qurat},
  booktitle = {Proceedings of the International Conference on Computing, Intelligence and Innovation (ICCII)},
  year      = {2026},
  url       = {https://github.com/ApatheticMioz/cancer-pathology-dl}
}
```

---

## ⚖️ License
This project is open-source and released under the [MIT License](LICENSE).
