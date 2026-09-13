# Cover Letter — Computers in Biology and Medicine

**Manuscript:** "All Dice, No Slice: Metric Artifacts, Data Leakage, and Task Interference in Multi-Task Computational Pathology"

**Authors:** Muhammad Abdullah Ali (corresponding), Muhammad Ibrahim Kiani, Muhammad Abdullah Aamir — Department of Data Science, FAST-NUCES, Islamabad, Pakistan

**Correspondence:** i232523@isb.nu.edu.pk

---

Dear Editors-in-Chief,

We are pleased to submit our manuscript, "All Dice, No Slice: Metric Artifacts, Data Leakage, and Task Interference in Multi-Task Computational Pathology," for consideration in *Computers in Biology and Medicine*.

## Summary of the work

This paper is a systematic replication and methodological audit of recently published multi-task pathology results. Rhanoui et al. reported a standard U-Net with lightweight encoders (VGG16, MobileNetV2) and static loss weighting achieving 82–90% accuracy and 97–99% Dice across four medical imaging modalities. We re-ran the reported protocol as a 26-run benchmark across four datasets (TCGA-LGG, PANDA, SIIM-ACR, PanNuke) and audited the underlying metrics, splits, and optimization recipe.

Our principal findings are:

1. **Leak-free benchmark.** Under strict patient/biopsy-level `GroupKFold` isolation, the claimed accuracies do not reproduce: 6-class PANDA grading spans 29.04–46.15% top-1 accuracy across all 12 configurations, against the claimed 88.0%.
2. **Metric-artifact analysis.** All four SIIM configurations converge to all-empty predictions yet report 77.74% Dice — exactly the lesion-free fraction of the validation split under the empty-mask convention that scores empty-over-empty as perfect agreement. We characterize this floor and its implications for sparse-lesion datasets.
3. **Matched ablation protocol.** Six strictly matched single-run ablation pairs, each reported with 95% Wilson score intervals, show that skip connections are a verified null result, that dynamic gradient balancing (GradNorm, α = 1.5) destabilizes classification (45.39% → 29.04%), and that removing Macenko stain normalization *improves* fine-grained classification (+5.51% PANDA, +2.68% PanNuke).
4. **Reporting recommendations.** We close with five recommendations for multi-task pathology studies: mandatory patient-level splitting, explicit Dice-convention reporting, Wilson intervals on matched ablations, multi-organ external validation, and systematic balancing-hyperparameter evaluation.

## Deliverables and availability

All training scripts, per-run JSON summaries, evaluation checkpoints, generated figures, and the complete results matrix with 95% Wilson score intervals are publicly available at https://github.com/ApatheticMioz/cancer-pathology-dl.

## Declarations

This manuscript has not been published previously and is not under consideration elsewhere. All authors have read and approved the submitted version. The authors declare no conflicts of interest. AI-assisted tools were used for language refinement and figure preparation only; all scientific content, experiments, data analysis, and conclusions are the sole work of the authors.

We believe this audit will be of broad interest to the CBM readership, as it identifies failure modes that can silently inflate reported performance in multi-task medical imaging studies and offers a reproducible protocol to prevent them.

Thank you for your consideration.

Sincerely,

Muhammad Abdullah Ali (on behalf of all authors)
