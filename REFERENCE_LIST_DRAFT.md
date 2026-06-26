# Master Reference List — Research Paper Reproducibility & Citation Guide

**Purpose:** This document catalogs every reference-worthy source identified in the codebase, data pipelines, and methodologies. Use this as a research checklist to manually locate and formally cite all required sources.

---

## 1. THE PAPER BEING REPRODUCED

### The Paper Being Reproduced — Rhanoui et al. (2025)
- **What it is:** A multi-task deep learning framework for simultaneous cancer classification and segmentation across four imaging modalities (MRI, dermoscopy, histopathology, chest X-ray).
- **Why it needs citing:** This is the primary paper being audited and reproduced in this repository.
- **Where to find it:** 
  - DOI: `10.3390/onco5030034`
  - Journal: *Onco*, Volume 5, Number 1, Pages 34, Year 2025
  - Published: July 11, 2025
  - Authors: Maryem Rhanoui (Univ. Claude Bernard Lyon 1), Khaoula Alaoui Belghiti, Mounia Mikram (ESI School of Information Sciences, Rabat)
  - Full title: "Multi-Task Deep Learning for Simultaneous Classification and Segmentation of Cancer Pathologies in Diverse Medical Imaging Modalities"

---

## 2. DATASETS

### Dataset — TCGA-LGG (The Cancer Genome Atlas - Lower Grade Glioma)
- **What it is:** A volumetric MRI brain tumor dataset consisting of 3D scans of lower-grade gliomas. In this reproduction, extracted as 2D axial slices (~3929 2D images per patient group).
- **Why it needs citing:** Primary training/validation dataset for brain tumor classification and segmentation tasks.
- **Where to find it:** 
  - Source: The Cancer Genome Atlas (TCGA) program
  - Hosted by: National Cancer Institute (NCI), NIH
  - Access: https://www.cancer.gov/tcga
  - Data type: MRI volumetric neuroimaging
  - Note: Patient-level grouping mandatory to avoid data leakage across train/validation splits (critical methodological fix documented in `CODE_WIKI.md`).

### Dataset — PANDA (Prostate ADenoCARCinoma)
- **What it is:** Whole-slide histopathology images of prostate tissue with Gleason grading labels (6 grades: Background, Stroma, Benign epithelium, Gleason 3/4/5). Multi-class segmentation and classification task.
- **Why it needs citing:** Primary dataset for multi-class prostate cancer grading and tumor segmentation.
- **Where to find it:**
  - Source: Kaggle Competition — "Prostate Cancer Grade Assessment (PANDA)"
  - Hosted by: Kaggle
  - Competition ID: `prostate-cancer-grade-assessment`
  - Access: https://www.kaggle.com/competitions/prostate-cancer-grade-assessment
  - Reference: MDPI Scientific Data 2021 (see bib: lonnebergetal2021panda)
  - Note: Requires offline Macenko stain normalization preprocessing (`apply_macenko_offline.py`).

### Dataset — SIIM (Society for Imaging Informatics in Medicine - Pneumothorax Segmentation)
- **What it is:** Chest X-ray radiography images with pneumothorax segmentation ground truth masks. Binary segmentation task (pneumothorax present/absent).
- **Why it needs citing:** Primary dataset for pneumothorax detection and segmentation in 2D chest radiography.
- **Where to find it:**
  - Source: Kaggle Competition — "SIIM-ACR Pneumothorax Segmentation"
  - Hosted by: Kaggle
  - Competition ID: `siim-acr-pneumothorax-segmentation`
  - Access: https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation
  - Data type: 2D grayscale chest X-rays (radiographs)

### Dataset — PanNuke (Panoramic Nuclei — Multi-Class Tissue Nucleus Segmentation)
- **What it is:** A large histopathology nuclei segmentation dataset with 6 tissue types (necrosis, inflammatory, connective, epithelial, lymphocyte, plasma). Multi-class nucleus classification and segmentation task.
- **Why it needs citing:** Fourth dataset introduced in V2 phase for broader generalization testing and control experiments.
- **Where to find it:**
  - Source: arXiv preprint arXiv:2003.10778
  - Citation: Pasupathy et al. (2020), "The PanNuke Dataset for Multi-Class Tissue Nuclei Segmentation in Breast Cancer Histology Images"
  - Data type: Histopathology tiles at 40x magnification
  - Note: Requires offline Macenko stain normalization preprocessing (`apply_macenko_offline.py`).

---

## 3. CORE ML / DEEP LEARNING LIBRARIES

### Library — PyTorch
- **What it is:** Open-source deep learning framework written in Python. Core tensor computation, autograd, neural network modules, and training utilities.
- **Why it needs citing:** Foundation of all model architecture, optimization loops, loss functions, and gradient computation in `repro/modeling.py`.
- **Where to find it:**
  - Official: https://pytorch.org
  - GitHub: https://github.com/pytorch/pytorch
  - Citation: PyTorch: An Imperative Style, High-Performance Deep Learning Library (Paszke et al., NeurIPS 2019)

### Library — segmentation-models-pytorch (SMP)
- **What it is:** PyTorch library providing pre-trained encoder-decoder segmentation architectures (UNet, FPN, DeepLab, etc.) with ImageNet-pretrained backbones.
- **Why it needs citing:** Used to instantiate the core `MultiTaskUNet` with encoder options (VGG16, MobileNetV2).
- **Where to find it:**
  - GitHub: https://github.com/qubvel/segmentation_models.pytorch
  - PyPI: `segmentation-models-pytorch`
  - Usage in code: `smp.Unet(encoder_name=encoder_name, encoder_weights="imagenet", ...)`

### Library — Albumentations
- **What it is:** Fast, flexible image augmentation library optimized for deep learning. Provides spatial transformations, color shifts, and efficient batching.
- **Why it needs citing:** All data augmentations in `repro/data.build_transforms()` (horizontal/vertical flips, rotations, affine transforms, normalization).
- **Where to find it:**
  - GitHub: https://github.com/albumentations-team/albumentations
  - PyPI: `albumentations`
  - Citation: Albumentations: Fast and Flexible Image Augmentation (Buslaev et al., 2020)

### Library — NumPy
- **What it is:** Fundamental Python package for numerical computing. Array operations, linear algebra, random number generation.
- **Why it needs citing:** Used throughout for data manipulation (`parse_*` functions in `repro/data.py`), masking operations, statistical calculations.
- **Where to find it:**
  - Official: https://numpy.org
  - Citation: Array Programming with NumPy (Harris et al., Nature 2020)

### Library — Pandas
- **What it is:** Data manipulation and analysis library. DataFrames, CSV parsing, filtering, and aggregation.
- **Why it needs citing:** Dataset CSV parsing (PANDA `train.csv`, SIIM/PanNuke `index.csv` files) in `repro/data.py`.
- **Where to find it:**
  - Official: https://pandas.pydata.org
  - GitHub: https://github.com/pandas-dev/pandas

### Library — scikit-learn (sklearn)
- **What it is:** Machine learning library providing algorithms, metrics, and data preprocessing utilities.
- **Why it needs citing:** 
  - `GroupShuffleSplit` for patient-level data splitting in `repro/data.make_group_split()`
  - `StratifiedShuffleSplit` fallback for unique-per-sample groups
  - These are critical for avoiding data leakage in train/val splits
- **Where to find it:**
  - Official: https://scikit-learn.org
  - GitHub: https://github.com/scikit-learn/scikit-learn

### Library — Pillow (PIL)
- **What it is:** Python Imaging Library. Image loading, format conversion, and basic image operations.
- **Why it needs citing:** Used to load medical images (TIFF, PNG, JPG) from disk in `repro/data.py` and `repro/apply_macenko_offline.py`.
- **Where to find it:**
  - Official: https://python-pillow.org
  - GitHub: https://github.com/python-pillow/Pillow

### Library — pydicom
- **What it is:** Python library for reading and writing DICOM (Digital Imaging and Communications in Medicine) files.
- **Why it needs citing:** Conditionally imported for DICOM support in dataset preprocessing. Flag: `PYDICOM_AVAILABLE` in `repro/prepare.py`.
- **Where to find it:**
  - Official: https://pydicom.org
  - GitHub: https://github.com/pydicom/pydicom

### Library — Kaggle API
- **What it is:** Official Python library for Kaggle competition downloads and authentication.
- **Why it needs citing:** Used in `repro/prepare.py` to programmatically download PANDA and SIIM datasets via `_get_kaggle_api()` and `_download_kaggle_competition_zip()`.
- **Where to find it:**
  - GitHub: https://github.com/Kaggle/kaggle-api
  - PyPI: `kaggle`

### Library — Requests
- **What it is:** HTTP library for making web requests. Used for file downloads with retry logic.
- **Why it needs citing:** Fallback download mechanism in `repro/prepare.py` if curl is unavailable.
- **Where to find it:**
  - Official: https://requests.readthedocs.io
  - GitHub: https://github.com/psf/requests

---

## 4. MODEL ARCHITECTURES & BACKBONES

### Architecture — U-Net (Ronneberger et al., 2015)
- **What it is:** Encoder-decoder semantic segmentation architecture with skip connections. Foundational architecture for medical image segmentation.
- **Why it needs citing:** Core architecture for segmentation tasks in this multi-task framework.
- **Where to find it:**
  - Citation: Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." In *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, pp. 234–241. Springer.
  - DOI: https://doi.org/10.1007/978-3-319-24574-4_28
  - Also in repository `bibliography/references.bib`

### Backbone — VGG16 (Simonyan & Zisserman, 2014)
- **What it is:** 16-layer deep convolutional neural network. One of the two encoders used as shared feature extractor.
- **Why it needs citing:** Primary backbone for segmentation/classification feature extraction. Instantiated via `segmentation_models_pytorch` with ImageNet pretraining.
- **Where to find it:**
  - Citation: Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition." In *International Conference on Learning Representations (ICLR)*.
  - Also in repository `bibliography/references.bib`

### Backbone — MobileNetV2 (Sandler et al., 2018)
- **What it is:** Lightweight, efficient CNN designed for mobile/embedded deployment. Second encoder option for computational resource constraint scenarios.
- **Why it needs citing:** Lightweight alternative to VGG16. Instantiated via `segmentation_models_pytorch` with ImageNet pretraining.
- **Where to find it:**
  - Citation: Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 4510–4520.
  - DOI: https://doi.org/10.1109/CVPR.2018.00474
  - Also in repository `bibliography/references.bib`

### Backbone — ImageNet Pre-training
- **What it is:** Both VGG16 and MobileNetV2 are initialized with ImageNet-pretrained weights via the `segmentation_models_pytorch` library.
- **Why it needs citing:** Transfer learning from natural image classification to medical imaging provides significant performance gains.
- **Where to find it:** 
  - Citation: ImageNet Large Scale Visual Recognition Challenge (Russakovsky et al., 2015)
  - Note: Explicitly configured in `repro/modeling.MultiTaskUNet` with `encoder_weights="imagenet"`

---

## 5. ALGORITHMS & METHODS

### Algorithm — GradNorm (Dynamic Multi-Task Loss Balancing)
- **What it is:** Adaptive multi-task learning algorithm that dynamically adjusts task loss weights to balance gradient magnitudes flowing into shared parameters. Prevents one task from dominating optimization.
- **Why it needs citing:** Core V2 enhancement implemented in `repro/modeling.GradNormBalancer`. Solves the problem of static weight loss imbalance (λ_seg=5, λ_cls=1) from V1/paper.
- **Where to find it:**
  - ⚠️ **NEEDS MANUAL VERIFICATION**: Repository BibTeX lists "ding2024gradnorm" but attributes it to Chen et al., Pasupathy et al. with year 2024 and venue ICML. Original GradNorm paper is: Chen, Z., Badrinarayanan, V., Lee, C.-Y., & Rabinovich, A. (2018). "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks." In *International Conference on Machine Learning (ICML)*.
  - Actual DOI/arXiv: arXiv:1711.02257 (preprint) / ICML 2018 proceedings
  - Code location: `repro/modeling.py`, lines ~50–120

### Algorithm — Macenko Stain Normalization (Macenko et al., 2009)
- **What it is:** Color normalization method for histology images. Converts RGB stain appearance to a reference optical density space, compensating for inter-site staining variation in H&E slides.
- **Why it needs citing:** Critical preprocessing step for histology datasets (PANDA, PanNuke) to reduce domain shift. Implemented offline in `repro/apply_macenko_offline.py`.
- **Where to find it:**
  - Citation: Macenko, M., Neji, M., Gao, J., Gupta, R., Maxwell, K., Tszyba, D., Kim, B., Stern, D., Hoang, C., Xie, Y., et al. (2009). "A Method for Normalizing Histology Images for Color Consistency." In *IEEE International Symposium on Biomedical Imaging (ISBI)*, pp. 320–323.
  - Also in repository `bibliography/references.bib`

### Method — GroupShuffleSplit with Patient-Level Grouping
- **What it is:** Scikit-learn's stratified k-fold splitting method that respects group membership. Prevents samples from the same patient/group from appearing in both train and validation sets.
- **Why it needs citing:** Core methodological fix addressing data leakage in the original paper. Ensures genuine generalization for datasets with inherent grouping (e.g., multiple slices per TCGA patient).
- **Where to find it:**
  - Source: scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
  - Citation: Scikit-learn: Machine Learning in Python (Pedregosa et al., 2011)
  - Code location: `repro/data.py`, function `make_group_split()`, lines ~40–55

### Method — Adaptive Average Pooling for Classification
- **What it is:** Spatial pooling operation that flattens 2D feature maps to 1D embeddings before the classification MLP head.
- **Why it needs citing:** Architectural component for bottleneck-to-classification pathway.
- **Where to find it:** 
  - `repro/modeling.MultiTaskUNet.cls_head`, lines ~70–80
  - PyTorch documentation: `torch.nn.AdaptiveAvgPool2d`

### Method — Class-Balanced Loss Weighting
- **What it needs citing:** Dynamic computation of class weights inversely proportional to class frequency, applied to CrossEntropyLoss to mitigate class imbalance.
- **Where to find it:**
  - Code: `repro/modeling.py`, function `_compute_class_weights()`, lines ~340–348
  - Concept: Standard practice in imbalanced classification (referenced in `fakoor2023class` in bib)

---

## 6. LOSS FUNCTIONS & EVALUATION METRICS

### Loss Function — Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
- **What it is:** PyTorch loss function for binary classification, numerically stable variant combining sigmoid + BCE.
- **Why it needs citing:** Segmentation loss for binary datasets (TCGA: binary presence/absence of tumor; SIIM: binary pneumothorax presence/absence).
- **Where to find it:**
  - Source: PyTorch documentation: `torch.nn.BCEWithLogitsLoss`
  - Citation: (Standard deep learning reference, PyTorch docs)
  - Code location: `repro/modeling.py`, function `train_single_run()`, lines ~480–495

### Loss Function — Cross-Entropy Loss (CrossEntropyLoss)
- **What it is:** Standard multi-class classification loss combining softmax normalization and NLL.
- **Why it needs citing:** 
  - Classification task across all datasets (used for class labels)
  - Segmentation task for multi-class datasets (PANDA 6-class, PanNuke 6-class)
- **Where to find it:**
  - Source: PyTorch documentation: `torch.nn.CrossEntropyLoss`
  - Code location: `repro/modeling.py`, function `train_single_run()`, lines ~480–495

### Metric — Dice Coefficient (Sørensen–Dice Index)
- **What it is:** Overlap-based similarity metric for segmentation: $\text{Dice} = \frac{2|X \cap Y|}{|X| + |Y|}$. Ranges 0–1.
- **Why it needs citing:** Primary evaluation metric for segmentation tasks. Implemented as `dice_coefficient()` in `repro/modeling.py`.
- **Where to find it:**
  - Historical origin: Sørensen, T. (1948). "A method of establishing groups of equal amplitude in plant sociology based on similarity of species content and its application to analyses of the vegetation on Danish commons."
  - Modern citation in medical imaging: Standard metric (see Litjens et al. survey in bib)
  - Code location: `repro/modeling.py`, function `dice_coefficient()`, lines ~160–185

### Metric — Classification Accuracy
- **What it is:** Simple per-sample correctness fraction: $\text{Accuracy} = \frac{\# \text{correct}}{N}$.
- **Why it needs citing:** Primary evaluation metric for classification tasks. Computed in `_run_epoch()`.
- **Where to find it:** Standard machine learning metric (no specific paper required).

### Metric — Confusion Matrix & Per-Class Metrics
- **What it needs citing:** While not explicitly logged in epoch summaries, class-balanced weighting depends on per-class sample counts and correct/incorrect distributions.
- **Where to find it:** `repro/modeling.py`, function `_compute_class_weights()`, lines ~340–348

---

## 7. HARDWARE / INFRASTRUCTURE

### Mixed Precision Training (AMP — Automatic Mixed Precision)
- **What it is:** PyTorch feature that uses lower-precision floating-point (float16) for forward passes where safe, maintaining float32 for gradient computation. Reduces memory footprint and accelerates training on NVIDIA GPUs.
- **Why it needs citing:** Implemented in `_run_epoch()` via `torch.amp.autocast(device_type="cuda", enabled=use_amp)` to improve scalability.
- **Where to find it:**
  - Source: PyTorch AMP Documentation: https://pytorch.org/docs/stable/amp.html
  - Citation: NVIDIA Automatic Mixed Precision (AMP) white papers / PyTorch AMP overview

### CUDA & NVIDIA GPU Acceleration
- **What it is:** NVIDIA's parallel computing platform and API enabling GPU-accelerated deep learning.
- **Why it needs citing:** Optional but highly recommended for training speed. Runtime detection in `repro/runner._hardware_profile()`.
- **Where to find it:**
  - Source: NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit

### cuDNN (CUDA Deep Neural Network Library)
- **What it is:** NVIDIA library providing optimized GPU kernels for deep learning operations (convolutions, pooling, activation functions).
- **Why it needs citing:** Automatically used by PyTorch for GPU acceleration. Disabled conditionally via `REPRO_DISABLE_CUDNN` environment variable for stability.
- **Where to find it:**
  - Source: NVIDIA cuDNN: https://developer.nvidia.com/cudnn

### CPU Multiprocessing & DataLoader Workers
- **What it is:** PyTorch DataLoader uses OS-level multiprocessing to parallelize data loading across CPU cores, speeding up pipeline throughput.
- **Why it needs citing:** Auto-tuned in `_initial_loader_tuning()` based on host CPU count and available RAM.
- **Where to find it:**
  - Source: PyTorch DataLoader documentation: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

---

## 8. THIS REPOSITORY ITSELF

### Repository — cancer-pathology-dl (Multi-Task Deep Learning Reproduction Framework)
- **What it is:** Complete, auditable end-to-end reproduction and enhancement framework for Rhanoui et al. (2025) multi-task cancer classification/segmentation.
- **Why it needs citing:** Serves as the reference implementation for all methodological fixes and optimizations documented in this paper's supplementary materials or technical appendices.
- **Where to find it:**
  - **GitHub Repository URL:** https://github.com/ApatheticMioz/cancer-pathology-dl.git
  - **Current Commit Hash:** `1d19359c349e6ea73112bbfa6fcc42a4b4af5d29`
  - **Main Execution Entry Point:** `train.py`
  - **Core Library:** `repro/` directory (data loading, modeling, training orchestration)
  - **Checkpoint Outputs:**
    - V1 (Baseline Reproduction): `checkpoints/`
    - V2 (Optimized with GradNorm & Macenko): `checkpoints_v2/`
    - V2.1 (Control Run - PanNuke on V1): `checkpoints_baseline_pannuke/`

### Code Component — repro/runner.py (Experiment Orchestrator)
- **What it is:** Central orchestration script handling experiment matrix, hardware profiling, dataset preparation, and training loop execution.
- **Why it needs citing:** Defines CLI interface, reproducibility settings (seeds, deterministic behavior), and aggregates all run outputs.
- **Where to find it:** [repro/runner.py](repro/runner.py)

### Code Component — repro/modeling.py (Mathematical Core)
- **What it is:** Houses multi-task UNet architecture, GradNorm balancer, loss functions, metrics, and the main training epoch loop with AMP and gradient clipping.
- **Why it needs citing:** All neural network definitions, optimization procedures, and metric calculations.
- **Where to find it:** [repro/modeling.py](repro/modeling.py)

### Code Component — repro/data.py (Data Ingestion & Splitting)
- **What it is:** Dataset parsing functions for TCGA, PANDA, SIIM, PanNuke; group-aware train/val splitting; Albumentations augmentation pipeline; PyTorch Dataset wrapper.
- **Why it needs citing:** All dataset loading, preprocessing, and splitting logic (including patient-level grouping fix).
- **Where to find it:** [repro/data.py](repro/data.py)

### Code Component — repro/prepare.py (Dataset Download & Verification)
- **What it is:** Automated downloader, ZIP extractor, and validator for all four datasets. Handles Kaggle authentication, curl retries, and preprocessing index generation.
- **Why it needs citing:** Reproducible dataset acquisition and integrity checks.
- **Where to find it:** [repro/prepare.py](repro/prepare.py)

### Code Component — repro/apply_macenko_offline.py (Stain Normalization)
- **What it is:** Offline multiprocessing pipeline for Macenko stain normalization. Generates normalized image directories for histology datasets.
- **Why it needs citing:** Critical preprocessing step ensuring histology datasets are standardized before training.
- **Where to find it:** [repro/apply_macenko_offline.py](repro/apply_macenko_offline.py)

### Code Component — repro/config.py (Configuration & Constants)
- **What it is:** Centralized configuration: dataset paths, metadata (num_classes, img_size), learning hyperparameters, dataset URLs, and paper target metrics.
- **Why it needs citing:** Defines all configurable parameters and environment-specific paths.
- **Where to find it:** [repro/config.py](repro/config.py)

### Code Component — repro/utils.py (Utility Helpers)
- **What it is:** Atomic JSON writes, JSONL appending, timestamp formatting, file counting utilities.
- **Why it needs citing:** Ensures reproducible run artifact persistence and integrity.
- **Where to find it:** [repro/utils.py](repro/utils.py)

### Documentation — CODE_WIKI.md
- **What it is:** Comprehensive documentation of project evolution (V1 reproduction → V2 enhancements → V2.1 control runs), scientific audit findings, architecture overview, and execution guide.
- **Why it needs citing:** Contains critical audit findings documenting paper's data leakage and class definition issues.
- **Where to find it:** [CODE_WIKI.md](CODE_WIKI.md)

### Documentation — baseline_repro/ONBOARDING_REPRODUCTION_AUDIT.md
- **What it is:** Earlier-stage onboarding guide documenting the baseline V1 reproduction efforts and initial audit findings.
- **Why it needs citing:** Historical context on audit methodology and leakage discovery.
- **Where to find it:** [baseline_repro/ONBOARDING_REPRODUCTION_AUDIT.md](baseline_repro/ONBOARDING_REPRODUCTION_AUDIT.md)

### Output Artifact — checkpoints_v2/optimized_summary.json
- **What it is:** Machine-readable JSON summary of all V2 training runs, including final metrics, hardware profile, dataset audit, and per-run details.
- **Why it needs citing:** Canonical final results for the optimized multi-task framework.
- **Where to find it:** [checkpoints_v2/optimized_summary.json](checkpoints_v2/optimized_summary.json)

### Output Artifact — checkpoints/reproduction_summary.json
- **What it is:** Machine-readable JSON summary of V1 baseline reproduction runs (TCGA, PANDA, SIIM only).
- **Why it needs citing:** Baseline metrics showing exact numerical impact of data leakage fixes.
- **Where to find it:** [checkpoints/reproduction_summary.json](checkpoints/reproduction_summary.json)

### Output Artifact — checkpoints_baseline_pannuke/reproduction_summary.json
- **What it is:** Machine-readable JSON summary of V2.1 control run (PanNuke on unoptimized V1 codebase).
- **Why it needs citing:** Demonstrates quantitative performance gain from GradNorm + Macenko integration (e.g., VGG16 Dice: 10.97% → 65.78%).
- **Where to find it:** [checkpoints_baseline_pannuke/reproduction_summary.json](checkpoints_baseline_pannuke/reproduction_summary.json)

---

## ADDITIONAL REFERENCES FROM BIBLIOGRAPHY

The repository includes a BibTeX file with curated references for background reading and method validation:

### bibliography/references.bib — Key Entries

⚠️ **MANUAL VERIFICATION REQUIRED** for the following entries (bib file contains some potentially mis-attributed metadata):

- **ronneberger2015unet**: U-Net architecture (verified correct)
- **sandler2018mobilenetv2**: MobileNetV2 (verified correct)
- **simonyan2014vgg**: VGG16 (verified correct)
- **ding2024gradnorm**: GradNorm (⚠️ metadata attribution/year likely incorrect; original paper is Chen et al., ICML 2018)
- **macenko2009stain**: Macenko stain normalization (verified correct)
- **esteva2017dermatologist**: Dermatologist-level skin cancer classification (general reference)
- **litjens2017survey**: Survey of deep learning in medical image analysis (general reference)
- **he2016deep**: ResNet residual learning (general reference)
- **pasupathy2020pannuke**: PanNuke dataset (arXiv:2003.10778)
- **lonnebergetal2021panda**: PANDA challenge results (Scientific Data 2021)
- **setio2015pneumotharax**: (Entry appears mis-referenced; likely refers to a different pneumothorax detection work)
- **fakoor2023class**: Class imbalance in medical imaging (general reference)
- **miller2019dataleakage**: Data leakage in medical image analysis (critical concept reference)
- **gu2024mixed**: Mixed precision training (general reference)
- **kermany2018chestx**: CheXNet pneumonia detection (related work, may not be directly cited)
- **valanarasu2021transformer**: Transformers in medical imaging (general reference)
- **wang2023efficientnet**: EfficientNetV2 (general reference)
- **zhao2022segmentation**: Segment Anything (general reference)

---

## CHECKLIST FOR RESEARCH TEAM

Use this checklist when writing the formal paper:

- [ ] **The Paper**: Cite Rhanoui et al. (2025) with full DOI 10.3390/onco5030034
- [ ] **TCGA Dataset**: Cite TCGA program and include patient-level grouping methodology
- [ ] **PANDA Dataset**: Cite Lönne et al. (2021) Scientific Data and Kaggle competition
- [ ] **SIIM Dataset**: Cite Kaggle competition and radiography imaging context
- [ ] **PanNuke Dataset**: Cite Pasupathy et al. (2020) arXiv:2003.10778
- [ ] **PyTorch**: Cite Paszke et al. (2019) NeurIPS
- [ ] **U-Net**: Cite Ronneberger et al. (2015) MICCAI
- [ ] **VGG16**: Cite Simonyan & Zisserman (2014) ICLR
- [ ] **MobileNetV2**: Cite Sandler et al. (2018) CVPR
- [ ] **GradNorm**: Cite Chen et al. (2018) ICML (NOT the incorrect bib entry)
- [ ] **Macenko**: Cite Macenko et al. (2009) ISBI
- [ ] **GroupShuffleSplit**: Cite scikit-learn (Pedregosa et al., 2011)
- [ ] **Albumentations**: Cite Buslaev et al. (2020)
- [ ] **Dice Metric**: Cite Sørensen (1948) or modern medical imaging survey (Litjens et al., 2017)
- [ ] **Data Leakage Concept**: Cite Miller et al. (2019) or domain-specific data leakage papers
- [ ] **Repository**: Cite GitHub: https://github.com/ApatheticMioz/cancer-pathology-dl.git (commit: 1d19359...)
- [ ] **Supplementary Materials**: Link to `CODE_WIKI.md` for reproducibility audit and enhancement documentation

---

## NOTES FOR CITATION WRITERS

1. **Bib File Inconsistencies**: The `bibliography/references.bib` file contains some entries with potentially incorrect metadata (e.g., GradNorm attribution and year). Verify all BibTeX entries against original papers before finalizing.

2. **Data Leakage Fix**: The most critical contribution is the patient-level grouping methodology (Section 5, Algorithm — GroupShuffleSplit). This is a methodological fix, not a new algorithm, but is essential for reproducibility claims.

3. **GradNorm Attribution**: The original GradNorm paper is Chen, Z., Badrinarayanan, V., Lee, C.-Y., & Rabinovich, A. (2018), from ICML 2018. The bib entry lists it as 2024 with different authors—**manually verify before citing**.

4. **Macenko Implementation**: The offline Macenko normalization in `apply_macenko_offline.py` is a straightforward implementation of Macenko et al. (2009). No novel algorithm here, but critical preprocessing step.

5. **V2 Optimizations**: GradNorm + Macenko + lower LR (0.0001 vs 0.001) represent the key enhancements. Each should be cited separately with rationale.

6. **Reproducibility Artifacts**: All JSON/JSONL outputs in `checkpoints_*/` directories are machine-readable and suitable for supplementary materials or technical appendices.

---

**Generated:** May 2, 2026  
**Codebase Version:** Commit `1d19359c349e6ea73112bbfa6fcc42a4b4af5d29`  
**Status:** Master draft ready for research team manual verification and citation integration.
