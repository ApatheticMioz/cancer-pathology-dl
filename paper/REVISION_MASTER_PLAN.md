# REVISION MASTER PLAN — "All Dice, No Slice" camera-ready revision

**Manuscript**: `paper/main.tex` (259 lines; compiles to exactly 10 pages, LNCS `llncs`).
**Target**: MICCAI / Springer LNCS camera-ready, strict 10-page limit, title
*"All Dice, No Slice: Metric Artifacts, Data Leakage, and Task Interference in Multi-Task
Computational Pathology"*.
**Inputs (frozen ground truth)**: `paper/audit_layer1_results.md` (26-run matrix ↔ checkpoints,
Wald/Wilson CI proof, SIIM invariant floor), `paper/audit_layer2_mathcode.md` (code AST:
losses, architecture, metrics mechanism, GradNorm variants, Macenko, protocol),
`paper/audit_layer3_tex.md` (paragraph-by-paragraph manuscript audit; defect IDs C1–C11),
`paper/audit_layer4_lit.md` (citation integrity + literature verification gates).
**Invariants**: (I1) every number/equation/citation grounded in artifacts or verified
literature; (I2) zero-net-expansion — `main.tex` must not grow a net line; compiled pages ≤ 10
(target exactly 10); (I3) academic register — no hyperbole, no colloquialism, no AI filler;
(I4) all edits via atomic LaTeX changes + compile gate per layer.

---

## PART I — Blocking verification gates (execute BEFORE any prose edit)

**V1 — Wilson CI recomputation (fixes C2).**
Write `scripts/compute_wilson_ci.py` (reuse `wilson_score_interval`,
`scripts/run_canonical_gradnorm_panda.py` L41–49): z = 1.96, **raw** accuracies/dice from
`checkpoints/summary_*.json` (not the rounded CSV), exact per-dataset val-n pinned by V2
(provisional: TCGA 778, PANDA 2104, SIIM 2135, PANNUKE 1567). Regenerate
`paper/paper_results_matrix_with_ci.csv`. Emit an old→new diff and re-derive overlap verdicts
for every pair quoted in prose: (08,25), (10,26), (10,23), (16,24), (17,20), (18,20).
Only prose whose verdict survives unchanged may keep its wording; all interval values update.

**V2 — Code greps (pin exact facts; kill all VERIFY-CODE items).**
1. Resolution: resize target in `src/data.py` (L2 audit says 256×256 for all datasets;
   tex L229 says 128×128 → align tex).
2. Split isolation: group key + splitter per dataset (`GroupKFold`/`GroupShuffleSplit` on
   patient_id/slide_id) — CRITICAL for SIIM and PanNuke (C8); scope the isolation claim to
   what the code does.
3. Exact val-n per dataset (V1 input) and the SIIM empty-mask count: verify E/n and the
   batch-mean value 0.7774375410222295 reproduce under 66×32 + 1×23 batching (L2 reports
   1,659/2,135 — note 1,659/2,135 = 0.77752, so pin the exact integer count and n; do not
   publish "77.70%" without reproducing it).
4. Epochs / batch size / seeds / weight decay / ImageNet pretraining in `src/training.py`,
   `src/models.py` (provisional: 50 epochs, batch 32, ImageNet weights).
5. Macenko: OD threshold (0.15?), `np.linalg.eigh(cov)` vs "SVD on pixels" wording; epsilon
   handling (tex Eq. 4).
6. Metrics: the empty-branch code (`union==0 → 1.0`), whether ε exists, what happens when
   GT empty ∧ pred non-empty; which function feeds `best_val_dice`.
7. Dataset N's: TCGA 3,929 / 110 patients / "three-sequence"; PanNuke 7,901; SIIM 12,047;
   PANDA 10,616.
8. Canonical probe log `logs/canonical_gradnorm_run18.log`: epoch count (15?) and final
   43.06 / 39.98 exact values.

**V3 — Literature verification (WebSearch; fixes all VERIFY-LIT items, audit_layer4 §3).**
κ range (Bulten), ">1,000 teams", inter-observer Gleason κ (candidates: Allsbrook 2001 et seq.),
GradNorm canonical specifics (α=1.5, separate optimizer, Adam lr 0.025, stop-gradient),
Macenko base-10 OD, Reinke naming ("Metrics Reloaded" vs cited pitfalls-guide title; empty-mask
discussion), UNI/CONCH author-title fix (both bib entries suspected conflated), Virchow author
list, Rhanoui metadata vs local `onco-05-00034-v3.md`, PanNuke/TCGA-LGG/SIIM dataset citations.
Fallbacks: if a claim cannot be verified, delete or weaken it per audit specs — never keep an
unverified number.

---

## PART II — Layered rewrite ledger (execution order; tex line numbers from current main.tex)

### Layer 1 — Table 1 & §3.1 (empirical core)
1. **CI columns/prose**: apply V1 outputs to L237, L240, L246, L251; relabel every interval as
   "95% Wilson score interval" (now true). Delete "(p > 0.05 …)" at L237 → "the intervals
   overlap substantially; a directional claim is not supported by single-run comparisons".
   [C2]
2. **"unweighted" → "static 5:1-weighted ($\lambda_{seg}{:}\lambda_{cls}=5{:}1$)"** at L57,
   L69, L227. [C3]
3. **"best reproducible performance" (L227)** → "the strongest reproducible results are the
   static $10^{-3}$ runs (Run 04: 46.15% Acc; Run 20: 44.08% Dice)". [C6]
4. **Resolution (L229)** per V2.1 (expect 256×256). [C7]
5. **§3.1 protocol completeness (L222)**: add epochs, batch size, seeds, ImageNet-pretrained
   encoders per V2.4; cite datasets at first mention (Layer-4 items below). [C11]
6. Table 1 cells: already artifact-true (L1); after V1, no CI columns exist in the table —
   none to change; verify caption's group descriptions unchanged.

### Layer 2 — Section 2 (mathematics)
1. **Eq. 1 (L81)**: replace single linear projection with the implemented two-layer head:
   $\hat{\mathbf{y}}_{cls} = \mathrm{Softmax}(\mathbf{W}_2\,\sigma(\mathbf{W}_1 \mathrm{GAP}(\mathbf{z}) + \mathbf{b}_1) + \mathbf{b}_2)$,
   σ = ReLU, hidden width 256, dropout 0.5 at training time (L2 §2). Fix "linear projection"
   prose at L79.
2. **Eq. 2–3 (L87–96)**: keep; clarify attribution — our replication optimizes CE/BCE only,
   a *documented deviation* from Rhanoui's stated $\mathcal{L}_{BCE}+\mathcal{L}_{Dice}$;
   simplify "non-differentiable or soft-Dice penalties" → "without any Dice-based penalty
   term". [L2 §1]
3. **Eq. 4 (L102)**: drop ε or state the `union==0 → Dice = 1.0` override explicitly (per V2.6).
4. **Eq. 5 (L110–112)**: replace conditional-indicator notation with the per-slice mean
   definition: $\overline{\mathrm{Dice}}_{all} = \frac{1}{N}\sum_i [\mathbf{1}(Y_i=\emptyset
   \wedge \hat{Y}_i=\emptyset)\cdot 1 + \mathbf{1}(Y_i\neq\emptyset)\cdot \mathrm{Dice}_i]$,
   ρ = fraction of empty-GT slices; rename "Zero-Score Assignment" → "empty-credit convention"
   (L107).
5. **Eq. 6 + L113–117 (C1 reframe)**: keep $\rho + (1-\rho)d_{fg}$ arithmetic as a labeled
   *analytic illustration* ($d_{fg}=77.74\%, \rho=0.78 \Rightarrow 95.10\%$); then state the
   empirical fact: all four SIIM runs predict empty masks for all epochs
   (`epoch_log.jsonl`), so the measured 77.74% **is** the empty-slice fraction floor
   (batch-mean realization), and the foreground Dice on positive slices is 0%. Delete
   "which collapse to 77.74% under clinical foreground-only evaluation" (L117) and replace
   with the correct direction. Fig. 2 caption (L122): recaption panel (b)/(c) accordingly
   (panel (c) = analytic curve).
6. **§2.3 (L126–141)**: keep joint formulation (verified); unify $w_i$-inside-norm notation
   with the exact code quote (V2.6 / L2 §4A); add one sentence contrasting canonical Chen et
   al. (separate weight optimizer, stop-gradient) pointing forward to the probe paragraph.
7. **§2.4 (L143–148)**: state implemented OD exactly (−ln, clip [1,255], I₀=255 ✓); per V2.5
   describe stain-vector extraction as implemented (covariance eigendecomposition, angular
   1st/99th percentiles) — drop "SVD on pixels" if that is not the code; "sub-micron" →
   "sub-cellular" (×2, incl. Fig. 1 caption L153).

### Layer 3 — Section 4 (discussion, clinical grounding, ablations)
1. **§4.1 (L229)**: per-encoder attribution of 87/98 and 88/99; verified κ range; participants
   count; inter-observer κ with citation (V3); "40–52%" → explicit derivation with assumptions
   or deletion [C9]; "unfeasible" → "infeasible"; resolution per V2.1.
2. **§4.2 (L231–240)**: update CI values (V1); PanNuke Dice split out as +12.79 [C10];
   keep null-result framing (L237) and "consistent with over-smoothing" hypothesis phrasing.
3. **§4.3 (L242–251)**: update CIs (V1); delete "Crucially," (L248); **replace the future-work
   clause (L251) with the canonical-probe paragraph (PART III, §C.1)** [C4]; PCGrad only with
   citation.
4. **Run-18 caveat**: prose at L246 correctly says Run 18 CI is Wilson — after V1 all are;
   no special-casing needed.

### Layer 4 — Introduction (L62–74)
Ranges 82–90 / 97–99 [C5]; ground the "lightweight alternative" sentence (cite the audited
study / an MTL reference); "multi-faceted clinical reasoning" → concrete phrasing; "teardown" →
"systematic methodological audit"; drop "exhaustive" (L67).

### Layer 5 — Conclusion (L253–254)
Scope "cannot achieve" to "not reproducible in our 26-run campaign under patient isolation and
artifact-free evaluation"; recommendation (3) → "95% Wilson score intervals"; recommendation
(1) scoped per V2.2; add reporting of the Dice convention used (foreground-only vs
empty-credit) — our SIIM floor is the demonstration.

### Layer 6 — Abstract, title, frontmatter (L44–60)
1. **Title (L44)** → "All Dice, No Slice: Metric Artifacts, Data Leakage, and Task Interference
   in Multi-Task Computational Pathology"; **\titlerunning (L45)** → "All Dice, No Slice:
   Metric Artifacts and Task Interference".
2. **Abstract rewrite** (word-for-word against Layers 1–5): fix ranges [C5]; fix evaluation
   description [C1] — new key sentence per PART III §C.2; "unweighted" [C3]; "vs 87–88% /
   98–99% claimed"; drop "exhaustive"/"rigorous"; concrete two-finding statement replacing
   "methodologically incompatible … clinically unfeasible"; keep 26-run/four-dataset/PANDA
   ranges (verified), null result, +5.51/+2.68.
3. Keywords (L59): optionally add "Data Leakage".
4. Authors/institute (L47–52): unchanged unless authors direct.

### Cross-cutting register pass (12 in audit_layer3 §12)
"trivially" ×3 → "by construction"; "exhaustive" ×3 → delete; plus items listed there.

---

## PART III — New content drafts (placeholders ⟨…⟩ filled by V2/V3)

**§C.1 — Canonical decoupled GradNorm probe (replaces the L251 future-work clause; ~9 lines,
paid for by the deleted clause + register-tightening):**

> \paragraph{Canonical decoupled GradNorm probe.}
> To separate the effect of the joint weight update from the canonical formulation, we
> additionally evaluated a decoupled variant of \textsc{GradNorm}~\cite{chen2018gradnorm} on
> PANDA$\times$VGG16 under identical raw-input conditions: the task weights are optimized by a
> dedicated \texttt{Adam} optimizer ($\eta_w = 0.025$) separate from the network optimizer
> (\texttt{Adam}, $\eta = 10^{-3}$), and network parameters are explicitly detached
> (stop-gradient) when the gradient-disparity loss $\mathcal{L}_{grad}$ is formed. The
> decoupled formulation remains stable — it avoids the $29.04\%$ collapse of the joint variant
> (Run 18) — yet plateaus at $43.06\%$ top-1 accuracy and $39.98\%$ Dice over ⟨EPOCHS⟩ epochs,
> below the static 5:1 baseline (Run 20: $45.39\%$ / $44.08\%$) and $44.94$ percentage points
> below the claimed $88.0\%$ accuracy. Dynamic gradient balancing, in either formulation, does
> not overcome the difficulty of patient-isolated 6-class grading; gradient-surgery methods
> (e.g., PCGrad~\cite{yu2020pcgrad}) and lower-asymmetry schedules ($\alpha \le 0.5$) remain to
> be evaluated.

**§C.2 — Abstract key sentence replacing "Under strict patient-level data isolation … patient
isolation" (L57):**

> Under strict patient-level data isolation (\texttt{GroupKFold}), the claimed classification
> accuracies are not reproducible, and the claimed Dice levels are attainable only under
> empty-mask-crediting conventions: in our own campaign, all four SIIM configurations converge
> to all-empty predictions yet still report $77.74\%$ macro Dice — exactly the empty-slice
> fraction — under the convention that scores empty-over-empty slices as perfect agreement.

**§C.3 — Protocol sentence for §3.1 (L222), placeholders per V2.4:**

> All encoders are ImageNet-pretrained; models train for ⟨EPOCHS⟩ epochs with batch size
> ⟨BATCH⟩ under fixed seeds ⟨SEEDS⟩.

**§C.4 — Bibliography changes**: cite `gamper2020pannuke` at L219; add TCGA-LGG dataset entry,
SIIM-ACR challenge entry (@misc, URL), optionally `yu2020pcgrad`; fix the two suspected
conflated foundation-model entries (`lu2024visual` → Chen et al., "A vision foundation model
for cancer pathology"; `lu2024multimodal` → Lu et al., "A visual-language foundation model for
computational pathology") per V3; remove `vahadane2016structure` (author bug, uncited) unless
cited at L148; verify `bulten2022artificial` exact title; brace-protect capitalization.

---

## PART IV — QA gates

Per layer: edit → `latexmk -pdf paper/main.tex` (0 errors) → page count (`pdfinfo`) ≤ 10 →
grep gates → commit. Grep gates: banned phrases ("trivially", "exhaustive", "teardown",
"unfeasible", "Crucially", "sub-micron", "delve", "pivotal", "multifaceted", "crucial
nuance", "navigating the complexit", "rigorous methodological framework"); every CI value in
prose matches regenerated CSV; "Wilson" appears only where true; no number in prose absent
from the frozen artifact register.
Final: `latexmk -C` clean rebuild; pages == 10; `bibtex` warning-free; Table 1 ↔ CSV final
diff; visual inspection of Fig. 1/Fig. 2 subfigure alignment and Table 1 layout (Phase-2
multimodal); recaptioned Fig. 2 checked against regenerated curve only if regeneration is
needed.

## PART V — Git plan (atomic, conventional)

1. `docs(paper): add audit artifacts and revision master plan` (the four audit files + this plan).
2. `fix(paper): recompute Wilson score CIs and correct interval labels` (V1 script + CSV + prose).
3. `fix(paper): correct empty-mask Dice artifact framing and SIIM evaluation description` (C1).
4. `fix(paper): report canonical decoupled GradNorm probe; align formulations with code` (C4 + Layer 2).
5. `fix(paper): ground clinical claims, dataset citations, and bib metadata` (V3 + Layer 4).
6. `refactor(paper): title, abstract, register pass, protocol completeness` (Layers 5–6).
Push to remote tracking branch on completion (per repo protocol).

## PART VI — Deferred / optional

Fig. 2 regeneration only if recaptioning is insufficient (panel (b) currently labels a
"foreground Dice 77.74%" that our artifacts contradict); hyperref link colors → black/dark for
print (cosmetic); Run 17/20 label hygiene lives in repo CSVs, not the manuscript — no tex action.
