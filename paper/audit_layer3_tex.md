# Layer 3 Forensic Audit: `paper/main.tex` Structure, Claims, and Rhetoric

Auditor: Lead Scientific Architect (Plan Mode). Method: line-by-line audit of `paper/main.tex`
(259 lines) against the frozen ground truth of `paper/audit_layer1_results.md` (results matrix,
checkpoints, CI provenance) and `paper/audit_layer2_mathcode.md` (code AST: losses, architecture,
metrics, GradNorm variants, Macenko, protocol). Status vocabulary: VERIFIED / MISMATCH /
UNGROUNDed / LANGUAGE / VERIFY-CODE / VERIFY-LIT.

---

## 0. Critical defects (ranked by scientific severity)

- **C1 — SIIM 77.74% is mischaracterized as a "foreground Dice" (MISMATCH, highest severity).**
  L2 proves: all four SIIM runs (05/06/11/12) predict all-zero masks for all 50 epochs
  (`checkpoints/epoch_log.jsonl`); under the implemented convention (union==0 → Dice=1.0) the
  macro score equals the empty-slice fraction — 1,659/2,135 ≈ 77.7%, realized as
  `0.7774375410222295` via unweighted batch-mean averaging (66×32 + 1×23 batches). Therefore:
  - L57 (abstract): "standard non-empty foreground Dice evaluation" — **wrong description of our
    own evaluation** for SIIM (empty-over-empty slices are scored 1.0, not discarded).
  - L113: "a baseline model achieving a modest foreground Dice of 77.74%" — wrong; 77.74 is the
    measured *macro artifact floor*, and our SIIM foreground Dice is ≈0 (all-empty predictions on
    positive slices score 0.0).
  - L117: "which collapse to 77.74% under clinical foreground-only evaluation" — wrong direction;
    under foreground-only evaluation our SIIM runs collapse to ≈0%, and 77.74% is what the
    empty-crediting convention reports.
  - L122 (Fig. 2 caption, panel b): "yielding the true clinical foreground Dice of 77.74%" —
    wrong label; the value is the aggregate artifact floor. Fig. 2(c) is an *analytic
    illustration* (rho grid with d_fg=0.7774 hardcoded, `scripts/generate_figures.py` L144–154)
    and must be captioned as such.
  - **Required reframe (strengthens the thesis)**: our own pipeline *reproduces* the artifact it
    critiques — all four SIIM configurations sit exactly on the empty-mask floor, demonstrating
    that the convention masks total negative-class collapse behind a stable 77.74%.
- **C2 — CI method mislabeled as Wilson (MISMATCH).** L1 proves 25/26 rows of
  `paper_results_matrix_with_ci.csv` are Wald intervals from
  `scripts/prepare_empirical_proofs.py` L76–104 with approximate n (TCGA 786 vs 778; SIIM 2409 vs
  2135; PANNUKE 1500 vs 1567; PANDA 2104 correct) computed on rounded accuracies; only Run 18 is
  a true Wilson interval (`run_canonical_gradnorm_panda.py` L41–49). Affected prose: L237
  ("95% Wilson confidence intervals", plus "(p > 0.05 …)" — an unperformed test), L240 (CIs
  quoted), L246 ("95% Wilson CI" — this one is Wilson, but the same label must not imply all
  are), L251 (CIs quoted). **Fix**: recompute true Wilson intervals for all 26 rows with exact
  per-dataset val-n, regenerate CSV + all quoted intervals, re-derive overlap/disjointness.
- **C3 — "Unweighted" baselines (MISMATCH, 3 sites).** L57, L69, L227 call Runs 03–04
  "unweighted"; they use static λ_seg:λ_cls = 5:1 (L1 args; L2 §1). Replace with "static
  5:1-weighted".
- **C4 — Canonical GradNorm probe omitted; framed as future work (MISMATCH).** L251: "future
  work should evaluate whether decoupled inner-loop optimizers … can stabilize" — but the probe
  exists and was run: `scripts/run_canonical_gradnorm_panda.py` (dual Adam: network lr=1e-3,
  weights lr=0.025; explicit `.detach()` of network parameter norms), result 43.06% Acc /
  39.98% Dice (`logs/canonical_gradnorm_run18.log`). Must be reported as an executed experiment
  (draft paragraph provided in `REVISION_MASTER_PLAN.md` §C.1). PCGrad mention (L251) is
  uncited — cite Yu et al. (2020) or drop.
- **C5 — Claimed ranges contradict Table 1's own Pub. columns (MISMATCH).** L57 and L65 say
  "86%–90%" accuracy / "95%–99%" Dice; Table 1 published targets are Acc 82–90 (SIIM VGG16 is
  82.0) and Dice 97–99. State 82–90% / 97–99% (or per-dataset values).
- **C6 — "Runs 03–04 … the best reproducible performance" (MISMATCH, L227).** Run 20 (45.39 /
  44.08) exceeds Run 03 (45.15 / 43.30); the campaign maxima are Run 04 Acc 46.15 and Run 20
  Dice 44.08. Reword.
- **C7 — Resolution contradiction (VERIFY-CODE).** L229 says "128×128 isolated patches"; L2
  audit of `src/data.py` L40–120 says 256×256 across all benchmarks (and L216 already says
  256×256 for TCGA). Grep `src/data.py` for the resize target; align L229 (and anywhere else).
- **C8 — Blanket isolation claim exceeds verified scope (VERIFY-CODE).** Abstract L57 "strict
  patient-level data isolation (GroupKFold)" — L2 verifies group-keyed splitting for TCGA and
  PANDA (`GroupShuffleSplit`/`GroupKFold` on patient_id/slide_id); SIIM and PanNuke isolation is
  NOT yet verified in code. Verify `src/data.py`; if ungrouped for SIIM/PanNuke, scope the claim
  ("patient/biopsy-level isolation on TCGA and PANDA; …on SIIM/PanNuke") — this affects
  Contribution 1 and Conclusion item (1).
- **C9 — Uncited load-bearing clinical claims (UNGROUNDed, L229).** (i) "inter-observer
  agreement kappa of only 0.65–0.75" — no citation; verify against literature (e.g., Allsbrook
  et al. 2001) and use the verified range. (ii) "corresponding to tile-level top-1 multi-class
  accuracies in the 40%–52% range" — our inference from κ, not a Bulten et al. result; present
  as an explicit derivation with stated assumptions or delete. (iii) ">1,000 participating
  teams" — verify exact count in Bulten et al. 2022.
- **C10 — PanNuke Dice characterized as "essentially unchanged" (MISMATCH, L240).**
  60.54% → 73.33% is +12.79 points. Report factually (PANDA Dice ~unchanged 31.34→31.57;
  PanNuke Dice increases +12.79).
- **C11 — PanNuke dataset never cited (L219) although `gamper2020pannuke` exists in the bib**;
  TCGA-LGG (L216) and SIIM-ACR (L218) dataset citations missing entirely (no bib entries).
  See `audit_layer4_lit.md`.

## 1. Frontmatter (L44–60)

| Line | Current | Status | Action |
|---|---|---|---|
| 44 | Title "On the Fragility of Multi-Task U-Nets …" | CHANGE | Replace with target title "All Dice, No Slice: Metric Artifacts, Data Leakage, and Task Interference in Multi-Task Computational Pathology". |
| 45 | `\titlerunning{On the Fragility …}` | CHANGE | "All Dice, No Slice: Metric Artifacts and Task Interference" (or shorter per LNCS headroom). |
| 47–52 | Authors, institute, emails | VERIFIED | No change required unless authors direct; `\authorrunning` "M. A. Ali et al." consistent. |
| 57 | "(86%–90%) … (95%–99%)" | MISMATCH | C5: 82–90% / 97–99% per Table 1 Pub. columns. |
| 57 | "standard non-empty foreground Dice evaluation" | MISMATCH | C1: describe the actually-implemented convention (empty-crediting macro Dice); foreground-only re-evaluation of SIIM gives ≈0. |
| 57 | "unweighted $10^{-3}$ baselines (Runs 03–04)" | MISMATCH | C3: "static 5:1-weighted ($\lambda_{seg}{:}\lambda_{cls}=5{:}1$)". |
| 57 | "vs $88.0\%$ and $99.0\%$ claimed" | PRECISION | 88.0/99.0 is the MobileNetV2 row; VGG16 target is 87/98 — "vs the claimed 87–88% / 98–99%". |
| 57 | "exhaustive 26-experiment replication" | LANGUAGE | Drop "exhaustive" (hyperbole; appears also L67, L254). |
| 57 | "we demonstrate that the published claims are methodologically incompatible with standard metric formulations and clinically unfeasible" | LANGUAGE/PRECISION | Vague; replace with the two concrete findings (Dice levels attainable only under empty-crediting conventions; accuracies not reproducible under patient isolation). |
| 57 | "We provide a rigorous methodological framework" | LANGUAGE | Drop self-praise adjective "rigorous". |
| 59 | Keywords | VERIFIED | Optional: add "Data Leakage" to mirror new title. |

## 2. Introduction (L62–74)

| Line | Current | Status | Action |
|---|---|---|---|
| 63 | "increasingly tasked with multi-faceted clinical reasoning" | LANGUAGE | Rephrase concretely (classification + segmentation co-prediction). |
| 63 | "remains widely explored as a lightweight alternative" | UNGROUNDed | Uncited generalization; ground with the audited study itself + an MTL reference, or rephrase as the specific setting under audit. |
| 65 | "reporting $86\%-90\%$ … $95\%-99\%$" | MISMATCH | C5. |
| 65 | "static loss weights (λ=5.0/1.0) and a high learning rate (η=10⁻³)" | VERIFIED | Matches L1 args/L2 §1. |
| 67 | "exhaustive 26-run replication campaign and methodological teardown" | LANGUAGE | "26-run replication campaign and systematic methodological audit" ("teardown" colloquial). |
| 69 | "collapses from the claimed $88.0\%$ … down to … $29.04\%$–$46.15\%$ … Runs 03–04 achieving 45.15–46.15 / 43.30–44.34" | MOSTLY VERIFIED | Numbers ✓ (L1); fix "unweighted" (C3); "collapses from the claimed 88.0%" overstates (87 for VGG16). |
| 70 | "or including background pixels, which trivially yields Dice ≡ 1.0" | PRECISION | Pixel-level background inclusion inflates but does not yield exactly 1.0; separate the two mechanisms. |
| 71 | "45.39% → 29.04% Acc" | VERIFIED | L1 Run 20 → Run 18. |
| 74 | GitHub URL + cite | VERIFIED | Matches `ali2026cancerpathologydl`. |

## 3. §2.1 Architectural formulation (L78–97)

| Line | Current | Status | Action |
|---|---|---|---|
| 81 (Eq. 1) | "$\hat{y}_{cls} = \mathrm{Softmax}(W_{cls}\cdot \mathrm{GAP}(z) + b_{cls})$" — "a linear projection" | MISMATCH | L2: head is GAP → Linear(d→256) → ReLU → Dropout(0.5) → Linear(256→K). Restate as two-layer MLP (or state both layers explicitly). |
| 83 | decoder + skips; sigmoid/softmax outputs | VERIFIED | Matches `MultiTaskUNet` (L2 §2). |
| 87 (Eq. 2) | joint weighted CE+BCE objective | VERIFIED | L2 §1. |
| 89–96 (Eq. 3) | Rhanoui stated BCE+Dice; our audit: CE-only training | VERIFIED + PRECISION | L2 §1 confirms no soft-Dice in backward pass. (a) Attribute clearly: *our replication* optimizes CE-only — a documented deviation from the published method; acknowledge as replication caveat. (b) "without non-differentiable or soft-Dice penalties" is awkward (soft-Dice is differentiable) — simplify to "without any Dice-based penalty term". |
| 97 | "DSC evaluated post-hoc" | VERIFIED | L2 §1. |

## 4. §2.2 Empty-mask Dice artifact (L99–124)

| Line | Current | Status | Action |
|---|---|---|---|
| 102 (Eq. 4) | Dice with +ε | PRECISION | ε-smoothing contradicts the union==0→1.0 override; either drop ε or state the override explicitly. |
| 104 | "two primary conventions … (e.g., Metrics Reloaded)" | VERIFY-LIT | Confirm the cited Reinke et al. paper actually discusses empty-mask conventions; the bib title is "Understanding metric-related pitfalls…" — do not call it "Metrics Reloaded" unless citing that paper. |
| 107 | "Zero-Score Assignment: Dice=1.0 if both empty" | LANGUAGE | Name is confusing (assigns 1.0, not 0.0); rename e.g. "empty-credit convention". |
| 110–112 (Eq. 5) | convex combination with indicator $\mathbf{1}_{\{\hat Y=\emptyset \mid Y=\emptyset\}}$ | PRECISION | Conditioning notation is abused; define as per-slice mean over indicators (empty∧correct → 1; non-empty GT → Dice_fg). |
| 113–116 (Eq. 6) | "modest foreground Dice of 77.74% … inflates to 95.10%" | MISMATCH (C1) | Keep the arithmetic as a *hypothetical illustration* (d_fg=77.74%, ρ=0.78 ⇒ 95.10%) but stop identifying 77.74% as our measured foreground Dice. |
| 117 | "predicting near-zero masks trivially yields >98%"; "collapse to 77.74% under clinical foreground-only evaluation" | MISMATCH (C1) | Under all-empty prediction the convention yields exactly the empty fraction (77.74% measured); foreground-only yields ≈0%. Rewrite per C1 reframe. |
| 122 (Fig. 2 caption) | panel (b) "true clinical foreground Dice of 77.74%"; panel (c) "automatically inflates" | MISMATCH (C1) | Recaption: (b) positive slice under all-empty prediction scores 0; (c) analytic curve. Check figure panels visually in execution (Phase-2 multimodal). |

## 5. §2.3 GradNorm (L126–141)

| Line | Current | Status | Action |
|---|---|---|---|
| 127 | "parameterized variant of GradNorm … Phase v2" | VERIFIED | Matches benchmark variant (L2 §4A). |
| 131 (Eq. 7) | $G_W^{(i)} = \|\nabla_W w_i \mathcal{L}_i\|_2$ | VERIFY-NOTATION | L2 quotes disparity as $|w_i G^{(i)} - \bar G [r_i]^\alpha|$ with $G^{(i)}=\|\nabla_W \mathcal{L}_i\|$; Eq. 7 folding $w_i$ inside the norm is equivalent for scalar $w_i$, but quote the code and unify notation. |
| 133 | log-weights μ=ln w, init 5.0/1.0, "updated jointly with the network parameters" | VERIFIED | L2 §4A (`nn.Parameter(torch.log(init))`, joint clip+step, single Adam). |
| 135–140 (Eq. 8–9) | disparity loss; α=1.5; relative rate | VERIFIED | L2 §4A. |
| 141 | "renormalized … $\sum_i w_i = 2$" | VERIFIED | L2 (normalize to 2.0). |
| — | Missing: explicit statement that the *canonical* Chen et al. formulation uses a separate weight optimizer and stop-gradient | GAP | Add one sentence here or (preferred) the probe paragraph in §4.3 (C4). |

## 6. §2.4 Macenko (L143–155)

| Line | Current | Status | Action |
|---|---|---|---|
| 146 (Eq. 10) | OD = −ln((I+ε)/I₀), I₀=255 | VERIFIED | L2 §5 (`-np.log(img/255)`, clip 1–255). |
| 148 | "natural logarithms rather than the canonical base-10" | VERIFIED | Correct vs Macenko 2009 (VERIFY-LIT the canonical form). |
| 148 | "extracted using singular value decomposition (SVD) on pixels exceeding an optical density threshold (>0.15)" | VERIFY-CODE | L2 quotes `np.linalg.eigh(cov)` (covariance eigendecomposition, angular 1st/99th percentile clipping) — equivalent plane but not literally "SVD on pixels"; verify the 0.15 threshold and restate exactly. |
| 148 | "chrominance low-pass filter, smoothing sub-micron chromatin textures" | LANGUAGE/PRECISION | "sub-micron" overreaches for 256-px tiles; "sub-cellular chromatin texture" is defensible. Mechanism is a hypothesis — keep "acts as" but tie to the +5.51/+2.68 ablation as "consistent with". |
| 153 (Fig. 1 caption) | "+5.51% on PANDA, +2.68% on PanNuke" | VERIFIED | L1 (Runs 10→23, 16→24). |

## 7. §3 + Table 1 (L157–222)

| Line | Current | Status | Action |
|---|---|---|---|
| 163–211 (Table 1) | 26 rows, Pub targets, Δ columns | VERIFIED | All cells match checkpoints at 2-dp (L1). CI columns pending Wilson recompute (C2). Caption's group descriptions consistent with JSON args (L1 map). |
| 216 | TCGA "2D $256\times256$ three-sequence MRI slices ($N=3{,}929$, 110 patients)" | PARTIALLY VERIFIED | 256² consistent with L2; N=3,929 / 110 patients / "three-sequence" unverified in artifacts — verify in `src/data.py` and cite the dataset (missing). |
| 217 | PANDA "$N=10{,}616$ tiles, patient-level biopsy isolation via GroupKFold" | VERIFIED | L1/L2; add `bulten2022artificial` (or data descriptor) at first mention. |
| 218 | SIIM "$N=12{,}047$ images" | PARTIALLY VERIFIED | N matches artifacts; isolation mechanism unverified (C8); dataset citation missing. |
| 219 | PanNuke "$N=7{,}901$ patches" | VERIFY + CITE | N unverified; `gamper2020pannuke` must be cited here (entry exists, currently never cited). |
| 222 | "fixed random seeds … Adam … constant LR … zero weight decay … no schedule" | PARTIALLY VERIFIED | L2 does not cover seeds/schedule — verify in `src/training.py`; add epochs (50) and batch size (32) per L2 §3 (verify globally, not just SIIM). |
| 222 | "In Phase v2 (Runs 07–12) … Macenko" | VERIFIED | Consistent with L1 JSON args (15/16 no_macenko=False; 13/14 raw). |

## 8. §4.1 PANDA anatomy (L226–229)

| Line | Current | Status | Action |
|---|---|---|---|
| 227 | ranges 29.04–46.15 / 28.94–44.34; 12 PANDA configs | VERIFIED | L1 (PANDA runs: 03,04,09,10,17,18,19,20,21,22,23,26 = 12 ✓). |
| 227 | "the unweighted static $10^{-3}$ baselines (Runs 03–04) … the best reproducible performance" | MISMATCH | C3 + C6: static 5:1; best Acc = Run 04 (46.15), best Dice = Run 20 (44.08). |
| 229 | "original authors reported 88.0% … 99.0%" | PRECISION | Attribute per-encoder (87/98 VGG16, 88/99 MobileNetV2). |
| 229 | "(>1,000 participating teams)" | VERIFY-LIT | Exact count in Bulten et al. 2022. |
| 229 | "quadratic weighted kappa 0.86–0.90" | VERIFY-LIT | Confirm external-validation κ range in Bulten et al. |
| 229 | "corresponding to tile-level top-1 … 40%–52%" | UNGROUNDed | Our inference — state assumptions explicitly or delete (C9). |
| 229 | "inter-observer agreement kappa of only 0.65–0.75 … due to continuous histological transitions" | UNGROUNDed | Needs citation + verified range (C9). |
| 229 | "$128\times128$ isolated patches" | VERIFY-CODE | C7: L2 says 256×256. |
| 229 | "unfeasible" | LANGUAGE | "infeasible". |

## 9. §4.2 Skip + Macenko ablations (L231–240)

| Line | Current | Status | Action |
|---|---|---|---|
| 234 | TCGA 93.32→94.34, 84.20→84.12, Δ+1.02/−0.08 | VERIFIED | L1. |
| 235 | PANDA 34.70→35.31, 31.34→28.94, Δ+0.61/−2.40 | VERIFIED | L1. |
| 237 | "95% Wilson confidence intervals … overlap substantially … (p > 0.05 …)" | MISMATCH + PRECISION | C2: relabel/recompute; delete the p-value claim (no test performed) — "intervals overlap substantially; a directional claim is not supported by single-run comparisons". |
| 237 | "zeroing the skip connections produces no detectable penalty" | VERIFIED (grounded in L2 mechanism: skips replaced by zeros, `src/models.py` L101–105) | Keep null-result framing; "A definitive directional claim would require multi-seed replication" is good discipline — keep. |
| 240 | Macenko +5.51/+2.68, disjoint CIs | VERIFIED | Numbers ✓; CI values update after Wilson recompute. |
| 240 | "Dice scores remain essentially unchanged (PANDA 31.34→31.57; PanNuke 60.54→73.33)" | MISMATCH | C10: PanNuke +12.79 is a substantial increase — report separately. |

## 10. §4.3 LR × GradNorm isolation (L242–251)

| Line | Current | Status | Action |
|---|---|---|---|
| 246 | Run 18 collapse 45.39→29.04 [27.14,31.02]; Dice 44.08→31.43; Δ−16.35/−12.65; non-overlapping vs [43.26,47.52] | VERIFIED | L1 (Run 18 CI is the only true Wilson row). Re-derive after recompute. |
| 248 | "matches … Run 09: 30.89/31.59" | VERIFIED | L1. "Crucially," opener — LANGUAGE (delete). |
| 251 | Run 17 vs 20 comparison, CIs [41.23,45.47] | VERIFIED | L1; update after recompute. |
| 251 | λ sweep conclusion "no weighting configuration can bridge the gap" | VERIFIED | Runs 19–22 max 45.39 < 87 claimed (L1). |
| 251 | "future work should evaluate whether decoupled inner-loop optimizers, gradient projection (e.g., PCGrad), or lower asymmetry schedules (α ≤ 0.5)…" | MISMATCH | C4: the decoupled probe was executed (43.06/39.98, stable). Replace with the probe paragraph; keep PCGrad/α-schedules as future work only if cited (Yu et al. 2020). |

## 11. Conclusion (L253–254)

| Line | Current | Status | Action |
|---|---|---|---|
| 254 | "cannot achieve the inflated 88%–99% metrics reported in recent multi-task pathology literature" | OVERBROAD/LANGUAGE | Scope to our campaign: "were not reproducible in our 26-run campaign under patient isolation and artifact-free evaluation"; "inflated" → keep only where the artifact mechanism is established. |
| 254 | five recommendations | VERIFIED against findings | (3) specify "95% Wilson score intervals"; (1) scope isolation claim per C8; add explicit mention of reporting the Dice convention used (foreground-only vs empty-credit) — the paper's own SIIM floor is the strongest argument for it. |

## 12. Register / cliché sweep (line → issue)

- L57 "exhaustive" ×1, L67 ×1, L254 ×1 — hyperbole; delete.
- L57 "an appealing paradigm" — filler; tighten.
- L63 "multi-faceted clinical reasoning" — rephrase.
- L67 "teardown" — colloquial; "systematic audit".
- L70/L113/L117 "trivially" ×3 — replace with "by construction".
- L148/L153 "sub-micron" ×2 — "sub-cellular".
- L229 "unfeasible" — "infeasible".
- L248 "Crucially," — delete.
- L57 "rigorous methodological framework" — drop "rigorous".
- Not present (verified clean): "delve", "pivotal", "multifaceted landscape", "crucial nuances", "navigating the complexities", "bogus", "shoddy", "disastrous".

## 13. Layout notes (page budget)

- Float tuning (L14–32) and 7.8pt bibliography (L28–32) already in place; document currently
  compiles to exactly 10 pages.
- Net additions planned: canonical-probe paragraph (§4.3), dataset citations (+2 bib entries:
  TCGA-LGG, SIIM-ACR; gamper entry already exists), protocol sentence (epochs/batch).
- Net removals paying for them: "future work" clause (L251), redundancy flagged above, tightened
  prose. Target: net-zero or negative line growth in `main.tex`.
- L34–40: hyperref colors (magenta/cyan) — consider muting to black/dark for print camera-ready
  (optional, cosmetic).
