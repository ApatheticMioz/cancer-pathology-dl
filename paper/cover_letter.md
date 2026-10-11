# Cover Letter — Medical Image Analysis

**Manuscript:** "All Dice, No Slice: Metric Degeneracy, Evaluation Pitfalls, and Optimization Dynamics in Multi-Task Medical Vision"

**Authors:** Muhammad Abdullah Ali (corresponding), Muhammad Ibrahim Kiani, Muhammad Abdullah Aamir — Department of Data Science, FAST-NUCES, Islamabad, Pakistan

**Correspondence:** i232523@isb.nu.edu.pk

---

Dear Editors-in-Chief,

We are pleased to submit our manuscript, "All Dice, No Slice: Metric Degeneracy, Evaluation Pitfalls, and Optimization Dynamics in Multi-Task Medical Vision," for consideration in *Medical Image Analysis* as an Original Research Article.

## Summary of the work

This paper provides a rigorous methodological audit and empirical reproduction of multi-task deep learning architectures commonly deployed for simultaneous medical image classification and segmentation. Rhanoui et al. (*Onco* 2025) reported standard U-Nets with lightweight encoders (VGG16, MobileNetV2) achieving 82–90% accuracy and 97–99% Dice across diverse medical imaging modalities. We conducted an exhaustive 27-configuration $\times$ 5-fold cross-validation campaign (135 total fold-runs) across four benchmark datasets (TCGA-LGG brain MRI, PANDA prostate biopsies, SIIM-ACR pneumothorax radiographs, and the PanNuke 19-tissue multi-organ benchmark), auditing the underlying metrics, splitting units, and optimization dynamics.

Our principal findings are:

1. **Resolution Limits & Task Formulation Mismatch.** In PANDA prostate biopsies, downsampling biopsy cores to 128x128 thumbnails destroys glandular architectural lumens. The baseline's reported 88% accuracy reflected Radboud morphological tissue textures (where background glass and stroma constituted 52.3% of tiles), whereas authentic clinical 6-class ordinal ISUP grading yields 28.21%–44.87% across all 12 configurations. Clinical grading from low-power thumbnails is fundamentally intractable without gigapixel multiscale attention pooling.
2. **Macroscopic Dice Degeneracy Floor.** In sparse imaging datasets (such as SIIM-ACR pneumothorax, with empty fraction $\rho=0.777$), standard macroscopic Dice behaves as a convex combination: $\overline{\text{Dice}}_{\text{all}} = \rho \cdot q_{\text{empty}} + (1-\rho)\overline{\text{Dice}}_{\text{fg}}$. All four SIIM models converge to all-empty predictions, yet mathematically report the ~77.7% degeneracy floor without segmenting any pathology.
3. **Multi-Task Optimization Dynamics.** Dynamic gradient balancing (GradNorm, $\alpha=1.5$) destabilizes fine-grained classification ($38.18\% \to 28.21\%$ on PANDA), skip connections offer no statistically detectable classification gain, and optical-density stain normalization attenuates chromatin texture contrast (+3.81% on PANDA, +2.75% on PanNuke when bypassed).
4. **Benchmark Integrity & Omission of ISIC 2018.** We document that ISIC 2018 Task 1 provided ground-truth segmentation masks for only 2,594 images while Task 3 contained 10,015 images (7,421 lacking segmentation masks). Because the published baseline lacked manifests or protocols for joint loss on unannotated samples, we evaluated the verified paired 19-tissue PanNuke benchmark instead.
5. **Verification Checklist.** We offer five concrete recommendations for multi-task medical vision: mandatory reporting of sample/patient grouping cardinalities, explicit mathematical definition of Dice empty-slice conventions, 95% bootstrap confidence intervals on matched ablations, multiscale whole-slide validation in computational pathology, and systematic tuning of multi-task balancing hyperparameters.

## Deliverables and availability

To ensure complete reproducibility, all training scripts, raw JSON summaries for all 27 configurations (135 fold-runs), evaluation checkpoints, generated figures, and the complete results matrix with 95% bootstrap confidence intervals are publicly available at https://github.com/ApatheticMioz/cancer-pathology-dl.

## Declarations

This manuscript has not been published previously and is not under consideration elsewhere. All authors have read and approved the submitted version. The authors declare no conflicts of interest. AI-assisted tools were used for language refinement and figure preparation only; all scientific content, experiments, data analysis, and conclusions are the sole work of the authors.

We believe this paper will be of substantial interest to the *Medical Image Analysis* readership, establishing essential verification standards for multi-task architectures intended for clinical translation.

Thank you for your consideration.

Sincerely,

Muhammad Abdullah Ali (on behalf of all authors)
