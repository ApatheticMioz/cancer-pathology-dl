# Issues in “All Dice, No Slice”

Page references refer to the supplied 26-page paper. The six findings below concern substantive problems in the paper’s reasoning or interpretation. They do not establish that the underlying experiment results are fabricated or numerically wrong. One unresolved result-reporting check appears separately at the end.

## 1. PANDA accuracy is compared across different classification tasks

**Location:** pp.3, 11–12, 14–16 and 21; Table 1 and Figure 4.

**Problem:** The cited study’s prostate labels are Background, Stroma, Benign epithelium and Gleason patterns 3/4/5. The paper instead predicts slide-level ISUP grades 0–5. These are different targets, even though both have six classes. The cited tissue-category accuracy of 87–88% and the paper’s ISUP accuracy of 28.21–44.87% therefore cannot be interpreted as an accuracy drop when reproducing the same task. The original label categories are documented in [Rhanoui et al., Table 2](https://www.mdpi.com/2673-7523/5/3/34).

**Why it matters:** This comparison supports the paper’s central claim of a replication gap. Changing the target prevents the numerical gap from identifying a failure to reproduce the reported performance. The label difference also does not, by itself, establish the exact implementation or cause of the cited accuracy.

**Fix:**

- Label the two results as **different tasks with non-comparable accuracies** in Table 1, Figure 4 and the text. Remove the PANDA numerical replication-gap interpretation unless a matched-label evaluation is provided.
- State the supported conclusion directly: “The cited study reports tissue-category classification, whereas our PANDA experiments evaluate ISUP grading; their accuracies are not directly comparable.”
- Remove the claim that thumbnail grading is universally intractable without MIL and that this representation necessarily produces the reported accuracy range. The experiment establishes performance for the tested models and inputs. A causal resolution claim requires a matched comparison across resolutions or representations.
- Apply the same comparability check to Dice: binary and multiclass segmentation scores require aligned targets and scoring definitions before a numerical gap is interpreted.

## 2. Empty-mask scoring does not establish why the cited study obtained 99% Dice

**Location:** pp.3 and 6; empty-mask analysis on p.7.

**Problem:** The paper attributes near-perfect published Dice to empty-mask credit. Its own collapsed SIIM predictions demonstrate a vulnerability of that scoring convention, but do not establish the cited study’s predictions, empty-mask convention or positive-case overlap.

The paper’s identity makes the distinction explicit. With negative prevalence `rho = 0.7772` and perfect empty predictions on negative cases, an all-empty predictor scores **77.72%**. Reaching **99%** would require positive-case Dice of approximately **95.51%**:

`D_positive = (0.99 - 0.7772) / (1 - 0.7772) ≈ 0.9551`.

This uses the paper’s prevalence as an illustration; the cited study’s prevalence is not established. Page 6 also makes an invalid inference from not training with Dice loss to high Dice scores being evaluation artifacts. Cross-entropy training can produce segmentations with high Dice.

**Why it matters:** The paper demonstrates that overall Dice can conceal foreground failure, but presents that demonstration as an explanation of another study’s score. Those are different evidentiary claims.

**Fix:** Replace the causal attribution with: “Our SIIM results show that empty-mask credit can obscure foreground segmentation failure. The cause of the cited 99% Dice cannot be determined without its predictions and evaluation protocol.” Remove the inference based on the absence of Dice loss. Keep the scoring identity and the observed collapse; report positive-case overlap and negative-case false positives alongside overall Dice.

## 3. Overlapping confidence intervals are treated as a test of no effect

**Location:** pp.15–16, 18–19 and 21; Figure 7.

**Problem:** The skip-ablation conclusion uses overlapping marginal confidence intervals to claim no statistically detectable classification benefit. Figure 7 explicitly assigns verdicts using a confidence-interval overlap test. Overlap does not establish that a paired difference is nonsignificant, and it does not establish equivalence. A paired comparison depends on uncertainty in the difference, including the relationship between the paired results.

Nonoverlap can indicate separation under appropriate assumptions; the specific error is using the overlap rule as a general decision procedure, particularly to infer no detectable benefit. Separately, resampling five fold accuracies does not account for all uncertainty from overlapping training sets or repeated model training. Increasing the number of bootstrap resamples does not increase the number of independent observations.

**Why it matters:** The current verdicts imply stronger statistical evidence than the analysis supplies.

**Fix:** Either report paired differences with a justified uncertainty method and clearly defined sampling unit, or make the figure descriptive and remove significance/equivalence verdicts derived solely from interval overlap. For the current skip results, use: “The observed classification differences were +0.02 percentage points on TCGA and −0.12 on PANDA; this analysis does not establish equivalence or test the significance of the paired differences.” Describe the fold bootstrap’s scope and limitations without presenting it as uncertainty over repeated independent training campaigns.

## 4. The GradNorm result is used to claim an unmeasured mechanism

**Location:** pp.1, 3, 17–18 and 21; Figure 6.

**Problem:** The matched comparison reports lower PANDA accuracy with the tested balancing recipe: 38.18% versus 28.21%, a 9.97-percentage-point decrease. That observation does not demonstrate that gradient-direction conflict caused the decrease or that `alpha = 1.5` was primarily responsible. Task-weight curves and gradient magnitudes do not measure whether task gradients point in conflicting directions. The on/off comparison does not isolate the effect of changing alpha.

**Why it matters:** The empirical result is useful, but the claimed optimization mechanism is not identified by the reported evidence. [GradNorm](https://proceedings.mlr.press/v80/chen18a.html) balances gradient magnitudes; a direction-conflict explanation requires additional measurements.

**Fix:** Retain the measured accuracy decrease and write: “Under the tested matched conditions, the GradNorm configuration achieved lower validation accuracy than the static-weight baseline.” Describe destructive gradient interference as a hypothesis. To retain a causal mechanism claim, measure task-gradient alignment during training; to attribute the effect specifically to alpha, compare alpha values under matched conditions. Broader claims of negative transfer also require matched single-task controls.

## 5. Macenko normalization is described as a spatial smoothing mechanism without evidence

**Location:** pp.10–11 and 18; Figure 3.

**Problem:** Page 10 describes stain projection as a low-pass filter that smooths subcellular textures, while p.18 treats smoothing as a hypothesis. Standard Macenko normalization estimates and transforms stain/color representations; that operation does not inherently perform spatial low-pass filtering. Image differences and higher accuracy without normalization do not identify smoothing as the cause. See the [original Macenko method](https://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf).

**Why it matters:** The paper moves from an observed preprocessing effect to a specific physical explanation that has not been demonstrated.

**Fix:** Describe the implemented color/stain transformation accurately and use hypothesis wording consistently in the text and caption. Keep the observed accuracy differences. Remove the asserted low-pass mechanism unless spatial-frequency or texture measurements demonstrate it for this implementation. A suitable statement is: “Normalization changed the image representation and was associated with lower classification accuracy in these matched experiments; the mechanism remains undetermined.”

## 6. ISIC image counts are used to infer an exact number of missing masks

**Location:** pp.14 and 20.

**Problem:** The paper subtracts 2,594 segmentation-training images from 10,015 classification-training images and concludes that exactly 7,421 classification images lacked segmentation ground truth. The [ISIC challenge paper, Sections 2.1 and 2.3](https://arxiv.org/pdf/1902.03368), confirms the two task sizes. Those counts alone do not establish their image-ID overlap or the availability of annotations from other sources. The exact missing-mask count is therefore unsupported by the evidence cited here; this is not a claim that 7,421 has been proven false.

**Why it matters:** The inferred count is used to justify the inability to reconstruct the skin experiment. Incomplete paired labels and an unrecoverable experimental protocol should be distinguished.

**Fix:** Verify the image-ID intersection and annotation availability before retaining the exact count. Otherwise, remove 7,421 and state: “The challenge reports different training-set sizes for segmentation and classification. The cited study does not provide enough information to reconstruct its paired-label cohort and training protocol.” Missing masks alone do not rule out joint training using a labelled subset or task-specific loss masking.

## Result-reporting check — not a confirmed numerical error

**Location:** p.11, Table 1; p.16, skip ablation; p.18, Macenko ablation.

**Observation:** Table 1 reports a TCGA Dice maximum of **88.85%**, whereas the skip discussion reports Run 08 at **89.22%**. It reports a PanNuke Dice minimum of **68.44%**, whereas the Macenko discussion reports Run 16 at **62.25%**.

**What remains uncertain:** These may use different scoring methods, aggregation units or checkpoint-selection rules. The numbers alone do not prove an incorrect result.

**Fix:** Trace these four values to their result records. State the Dice definition, aggregation unit and checkpoint rule beside each result. If the values measure the same quantity over the same run set, correct the table or discussion from those records. If they measure different quantities, label the distinction explicitly and avoid presenting them as a single comparable series. Identify Figure 5’s present-class per-tile Dice separately where it differs from the summary estimators.
