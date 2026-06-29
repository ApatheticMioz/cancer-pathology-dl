

Department of Data Science
School of Computing, FAST-NUCES
Course: CS-4112 – Deep Learning
## Instructor: Dr. Zohair
## Assignment 1: Deep Learning Research Proposal
Supervised Deep Representation Learning for
Simultaneous Multi-Task Classification and
Segmentation in Histopathology

## Prepared By:
M. Abdullah Ali  Roll: 23i-2523  i232523@isb.nu.edu.pk
Ibrahim KianiRoll: 23i-2536  i232536@isb.nu.edu.pk
Abdullah Aamir  Roll: 23i-2538  i232538@isb.nu.edu.pk
## Date: March 2, 2026
## 2

## 1. Problem Definition
## 1. Task Description
- Input: High-dimensional numerical tensors representing256×256 pixel RGB patches extracted
from Hematoxylin and Eosin (H&E) stained whole-slide tissue images.
- Output: The model simultaneously predicts two components: a discrete categorical probability
distribution (for classifying 19 tumor/tissue types) and a dense spatial matrix (a pixel-wise
binary mask for segmenting 5 nuclear categories).
- Learning Type: Supervised Learning. According to Goodfellow’s Deep Learning (Chapter
5), supervised learning algorithms experience a dataset containing features, where each example
is explicitly associated with a label or target [10]. Here, the network learns to map inputs to
both categorical labels and exact segmentation masks using explicit ground-truth annotations.
- Motivation and Relevance
- Real-World Importance: The manual analysis of histopathology slides is a critical bottleneck
in modern oncology. It is subjective, operator-dependent, and time-consuming. Automating this
process to simultaneously classify cancer and segment cell structures can drastically accelerate
tumor grading and improve patient safety.
- Appropriateness of Deep Learning: Histopathological images are incredibly complex. A
single256×256 RGB image patch comprises 196,608 dimensions. Navigating this vast space
via classical machine learning and manual feature engineering (e.g., hand-crafting morphological
cell rules) is practically impossible due to the curse of dimensionality, a concept explicitly
highlighted by Goodfellow [10].  We strictly require representation learning—using deep
networks to autonomously discover complex, hierarchical biological patterns directly from raw
pixels.
## 3. Clear Research Questions
- “Can a supervised multi-task deep neural network leverage shared learned representations to si-
multaneously outperform single-task baseline models in tissue classification and nuclear segmen-
tation without increasing computational overhead?”
- Literature Review (10 papers)
## 2.1. Overview
The trajectory of computational pathology strictly aligns with the paradigm shift described in
Goodfellow’s introduction to machine learning: transitioning from traditional, rule-based heuristics
to deep representation learning [10]. Recent advancements heavily favor multi-task deep networks
designed to extract shared structural and categorical features from complex medical images simul-
taneously, establishing powerful inductive biases.
## 1

2.2. Detailed Review of 3 Baseline (2024+) Papers
Baseline 1: Towards a general-purpose foundation model for computational pathology
(Chen et al., 2024) [1]
- Problem and Data: Addresses the fragmentation of pathology feature extractors using a
massive pre-training dataset of whole-slide images to create a unified foundation model.
- Deep Model Idea: A massive self-supervised neural network (Transformer) that learns gen-
eralized visual tissue representations from raw images, which are then fine-tuned for supervised
classification.
- Training Setup: Gradient-based optimization over large compute clusters, utilizing a discrim-
inative loss for multi-class diagnostics.
- Results and Metrics: Achieved state-of-the-art accuracy and F1 scores across 30 downstream
diagnostic subtyping tasks.
- Strengths: Exhibits incredible generalizability and robust feature representation.
- Weaknesses or Gaps: It functions fundamentally as a single-task classification engine.  It
ignores dense pixel-wise spatial segmentation, requires prohibitive computational limits, and
does not leverage the structural benefits of multi-task learning.
Baseline 2: Multi-Task Deep Learning for Simultaneous Classification and Segmenta-
tion of Cancer Pathologies... (Rhanoui et al., 2025) [2]
- Problem and Data: Targets the isolation of classification and segmentation in digital oncology
across diverse imaging modalities.
- Deep Model Idea: A deep neural network using an encoder-decoder framework to extract
shared representations, before branching into separate hidden-layer pathways for task-specific
outputs.
- Training Setup: Multi-loss gradient optimization combining cross-entropy and spatial overlap
metrics.
- Results and Metrics: Substantial precision and intersection-over-union (IoU) improvements
over isolated models.
- Strengths: Strongly validates task synergy, proving that segmentation pathways enhance clas-
sification features.
- Weaknesses or Gaps: Relies on static loss weights.  It does not dynamically balance the
gradients of the two tasks, risking a scenario where the classification gradients overpower the
segmentation pathways.
## 2

Baseline 3: Joint multi-task learning improves weakly-supervised biomarker prediction
in computational pathology (El Nahhas et al., 2024) [3]
- Problem and Data: Predicts localized biomarkers under weak supervision (where only image-
level labels are available).
- Deep Model Idea: Utilizes auxiliary feature extraction networks to force attention on local
tissue architecture alongside global biomarker prediction.
- Training Setup: Gradient-based training mapped to weakly-supervised loss formulations.
- Results and Metrics: Superior Area Under the Curve (AUC) by balancing primary and
auxiliary task learning.
- Strengths: Excellent theoretical algorithmic task balancing.
- Weaknesses or Gaps: Fundamentally limited by the lack of exhaustive pixel-wise ground
truth matrices, capping its structural precision compared to models trained on fully supervised
dense datasets.
2.3. Short Summaries of 7 Additional Papers
Papers that introduce the problem or dataset:
- Gamper et al. (2019) [4] introduces the PanNuke dataset, which provides the foundational
multi-task challenge of simultaneous pan-cancer histology segmentation and classification used
in this proposal.
- Srinidhi et al. (2021) [11] supplies a comprehensive survey of deep learning in histopathology,
heavily informing our preprocessing, stain normalization, and evaluation standards.
Papers that show other deep learning approaches in the same domain:
- Graham et al. (2023) [5] demonstrates a methodological predecessor for multi-task learning
in histology, proving that simultaneous segmentation and classification yield superior inductive
biases.
- Dosovitskiy et al.  (2021) [8] outlines the foundational theory for self-attention in vision,
allowing networks to process image patches sequentially for better global context.
- Ronneberger et al. (2015) [9] is the foundational biomedical segmentation paper, proving
that complex spatial reconstruction can be achieved by routing earlier representations directly
into deeper hidden layers.
Papers that discuss evaluation metrics or regularization techniques:
## 3

- Goodfellow et al. (2016) [10] provides the foundational textbook definitions for representation
learning, addressing the curse of dimensionality and formalizing supervised learning.
- Caruana (1997) [6] defines the mathematical basis of multitask learning, proving that shared
representations strictly improve network generalization.
- Chen et al. (2018) [7] introduces GradNorm, the dynamic gradient normalization mathemat-
ical technique we utilize to balance dual losses during simultaneous training.
## 3. Proposed Deep Learning Approach
- High-Level Idea
We propose to use a deep neural network that takes normalized pixel features as input and passes
them through several hidden layers to learn complex biological patterns.  The network then bi-
furcates: one pathway outputs class probabilities for the underlying tissue type, while the other
pathway outputs a spatial reconstruction matrix indicating cell boundaries.
## 2. Input Representation
The histopathological input data is represented numerically as continuous multidimensional arrays.
Specifically, the images are parsed as256×256 grids of basic numeric features with 3 color channels
(RGB). Prior to ingestion by the network, these numerical matrices are strictly normalized to
standard uniform scales to ensure stable optimization.
- Network Outline (Verbal)
The architecture utilizes a deep neural network consisting of several hidden layers configured to share
representations between the two tasks. Crucially, we utilize a non-linear activation function (specifi-
cally, a function that introduces non-linearity, such as ReLU) to ensure the network can capture the
highly complex morphology of cancer cells. Learning is executed strictly through gradient-based
optimization. The classification pathway utilizes categorical cross-entropy loss, while the spatial
reconstruction pathway uses a composite reconstruction loss. We will implement dynamic gradient
weighting to ensure neither task overpowers the other.
- Why this is Suitable
This design explicitly leverages Goodfellow’s core principles: deep networks can autonomously learn
complex, high-dimensional functions (representation learning) without requiring impossible hand-
crafted biological rules [10].  By sharing hidden layers across two tasks, the network can share
statistical strength. The segmentation requirement forces the model to encode fine cellular struc-
tures, which directly improves the mathematical representation utilized by the classification head.
## 4. Dataset Description
- Source of Data
This proposal utilizes the publicly available PanNuke dataset, an open-source pan-cancer histology
repository curated by Gamper et al. (2019) [4].
- Size and Structure
## 4

The dataset contains precisely 7,904 independent image samples.  For each sample, the features
are numeric256×256×3 RGB pixel arrays.  Being a strictly supervised multi-task paradigm,
each sample has two targets: a discrete classification label corresponding to 19 distinct human
tissue types, and exhaustive spatial pixel masks categorizing 5 distinct nuclear classes (neoplastic,
non-neoplastic epithelial, connective, inflammatory, and dead).
## 3. Preprocessing
Numeric features will undergo Macenko stain normalization and standard scaling to normalize
pixel intensity distributions. Textural masks are mathematically thresholded into discrete binary
boundaries to prevent multi-class pixel overlap during reconstruction.
- Why this Dataset Fits the Problem
PanNuke is uniquely suitable because it provides both slide-level semantic labels and exhaustive
pixel-level annotations natively. It provides a dense, multi-modal ground truth across nearly 8,000
high-dimensional examples, which is mathematically sufficient for deep representation learning with-
out resulting in immediate overfitting.
## 5. Evaluation Plan
## 1. Metrics
The dual nature of the network requires specific metrics for each pathway:
- Classification Metrics: We will utilize Accuracy and the Macro F1-score.  Due to the
high class imbalance among the 19 tissue types in PanNuke, the Macro F1-score is mandatory
to ensure underrepresented cancers are evaluated rigorously.
- Segmentation Metrics: We will employ the Dice Coefficient (to measure raw spatial over-
lap) and the Aggregated Jaccard Index (AJI). AJI is explicitly chosen because it severely
penalizes models that erroneously merge distinct adjacent cells into single continuous blobs.
## 2. Experimental Setup
The data will be strictly divided into a 70% training, 15% validation, and 15% hold-out test set. The
basic training plan will run gradient-based optimization across epochs until validation loss stabilizes,
utilizing explicit early stopping protocols to prevent overfitting. Hyperparameters requiring tuning
include the learning rate, the number of shared hidden layers, and batch size. Due to strict hardware
limits (a single 10 GB VRAM GPU), we will rely heavily on mixed precision and small batch sizes.
## 3. Baseline Comparison
We will compare our proposed model’s metrics (Accuracy, F1, Dice) against Baselines 1, 2, and
- We will reproduce or approximate these baseline models using identical PanNuke dataset splits
and similar normalization preprocessing. We do not need to reproduce the exact massive parameter
counts of Baseline 1; instead, a reasonable implementation consistent with each paper’s functional
description, scaled to 10 GB VRAM limits, will serve as the fair comparative baseline.
## 5

## 6. Expected Outcomes
## 1. Performance Expectations
We expect our deep neural network, augmented by dynamic multi-task loss balancing, to strictly
exceed the Macro F1-score of single-task counterparts (Baseline 1) on the exact same dataset. The
shared spatial representation forced by the segmentation pathway provides an inductive bias that
a standalone classification network lacks.
- Learning and Understanding Outcomes
We expect to validate the core deep learning concept of representation learning versus hand-crafted
features by mathematically demonstrating that deeper shared hidden layers generate superior spa-
tial feature maps.  We will also heavily explore the network’s sensitivity to gradient-balancing
hyperparameters.
## 3. Potential Limitations
The primary limitation is computational capacity.  Strict 10 GB VRAM limits will necessitate
architectural simplifications compared to the original massive baseline architectures, potentially
imposing an artificial ceiling on the model’s raw generalization performance. Limited dataset size
may also restrict generalization to poorly stained out-of-distribution clinical samples.
## 6

## 7. Deliverables & Reference List
As per the assignment requirements, a modular implementation of the proposed deep neural network
will be submitted. Below is the comprehensive list of all 10 research papers utilized to formulate
this proposal.  To prevent redundancy between the mandated "Paper List Document" and the
"References" section, they have been merged. The three mandated 2024+ baseline models and
their respective roles are explicitly highlighted in bold text.
[1] [2024+ Baseline 1] Chen, R. J., Ding, T., Lu, M. Y., Williamson, D. F., et al. (2024).
Towards a general-purpose foundation model for computational pathology. Nature Medicine,
## 30(3), 850–862.
[2] [2024+ Baseline 2] Rhanoui, M., et al. (2025). Multi-Task Deep Learning for Simultaneous
Classification and Segmentation of Cancer Pathologies in Diverse Medical Imaging Modalities.
## Onco, 5(3), 34.
[3] [2024+ Baseline 3] El Nahhas, O. S. M., et al. (2024). Joint multi-task learning improves
weakly-supervised biomarker prediction in computational pathology. Medical Image Computing
and Computer Assisted Intervention – MICCAI 2024. Springer.
[4] [Ancillary - Dataset] Gamper, J., Alemi Koohbanani, N., Benet, K., Khuram, A., & Rajpoot,
N. (2019). PanNuke: an open pan-cancer histology dataset for nuclei instance segmentation
and classification. European Congress on Digital Pathology.
[5] [Ancillary - Predecessor] Graham, S., et al. (2023). One model is all you need: Multi-task
learning enables simultaneous histology image segmentation and classification. Medical Image
## Analysis, 83, 102685.
[6] [Ancillary - Theory] Caruana, R. (1997). Multitask learning. Machine learning, 28(1), 41–75.
[7] [Ancillary - Regularization] Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. (2018).
GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks.
International Conference on Machine Learning (ICML).
[8] [Ancillary - Theory] Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Trans-
formers for Image Recognition at Scale. International Conference on Learning Representations
## (ICLR).
[9] [Ancillary - Predecessor] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolu-
tional Networks for Biomedical Image Segmentation. Medical Image Computing and Computer-
Assisted Intervention – MICCAI 2015.
[10] [Ancillary - Text/Theory] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning.
MIT press.
[11] [Ancillary - Survey] Srinidhi, C.L., Ciga, O., & Martel, A.L. (2021). Deep Neural Network
Models for Computational Histopathology: A Survey. Medical Image Analysis, 67, 101813.
## 7