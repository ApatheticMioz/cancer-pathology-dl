# Peer-Review Audit & Patch Plan
# Manuscript: "On the Fragility of Multi-Task U-Nets in Computational Pathology"
# Date: 2026-08-27 · Auditor: goose (full-repo audit of paper/main.tex, paper/*.csv, run_all_experiments.sh, src/, docs/)

## 0. Provenance of the feedback (read this first)

The reviewer's feedback is a mixture of (a) observations made against an OLDER
version of the paper/table, and (b) issues that DO exist in the current
main.tex. The repo contains multiple vintages of the same data:

| Artifact | Vintage | Status |
|---|---|---|
| `paper/paper_results_matrix.csv` (+ `_with_ci`) | FINAL (timestamps 2026-08-25/26) | Source of truth; all 26 values match `main.tex` Table 1 exactly |
| `paper/paper_results_latex_table.txt` (aggregator output) | STALE layout | Rows ordered by ablation type, NO run-# column; no-skip rows are at the TOP (positions 1-2); last two rows (25-26) ARE the No-Macenko runs (PANDA 40.21, PanNuke 99.36). This is where the reviewer's "rows 25-26 are No-Macenko" came from. Also mislabels the 43.35 run as "(α=1.5)" with an empty GradNorm cell. |
| `docs/technical_diagnostic_report.md` | EARLIER round | Different run map and numbers (e.g., PANDA "37.1-42.2%", "VGG16 collapse 10.97/17.40", run 17 = GN run) |
| `paper/scientific_review_and_conclusions.md` §D | internal review | Contains the OLD "skip paradox" framing (46.15/44.34 → 35.31/28.94, Δ = -10.8/-15.4) — this is the source of the reviewer's item 1 |
| `run_all_experiments.sh` | current | Authoritative run ordering (01-26) and CLI flags |

The current `main.tex` Table 1 (explicit #01-26 column) already fixed the
row-identity problem; the current §4.2 already uses the matched PANDA
counterpart (Run 10, not Run 04). But NEW, verifiable problems exist (see §2
below): **Run 18 is labeled a GradNorm run in the paper but provably was not**,
and **Run 17 (and 19-22) are mislabeled regarding Macenko / phase**.

---

## 1. Feedback item 1 — "Skip Connection Paradox" (§4.2)

### Verified numbers (all from the final CSV)
| Comparison | Acc | Dice | Acc 95% CI overlap? |
|---|---|---|---|
| TCGA: Run 25 (no-skip, v2) vs Run 08 (v2) — MATCHED (1e-4, GN, Macenko) | 94.34 vs 93.32 (+1.02) | 84.12 vs 84.20 (−0.08) | [92.72-95.96] vs [91.57-95.07] → OVERLAP |
| PANDA: Run 26 (no-skip, v2) vs Run 10 (v2) — MATCHED (1e-4, GN, Macenko) | 35.31 vs 34.70 (+0.61) | 28.94 vs 31.34 (−2.40) | [33.27-37.35] vs [32.67-36.73] → OVERLAP |
| PANDA: Run 26 vs Run 04 (v1 naked) — the OLD/confounded comparison the reviewer saw | 35.31 vs 46.15 (−10.84) | 28.94 vs 44.34 (−15.40) | — (LR, GN and Macenko all differ) |

### Opinion
- The reviewer is RIGHT that comparing Run 26 to Run 04 confounds LR (1e-4 vs 1e-3),
  GradNorm and Macenko; the 10.8/15.4-point "collapse" is not a skip effect.
  (That exact framing lives in `scientific_review_and_conclusions.md` §D and
  evidently in the draft the reviewer read.)
- The current main.tex §4.2 ALREADY compares Run 26 vs Run 10 (matched) — so the
  citation itself no longer matches the reviewer's quote. However, even the
  matched comparison does NOT support the current text:
  * "+0.61 Acc / −2.40 Dice" on PANDA and "+1.02 Acc / −0.08 Dice" on TCGA are
    single-run differences whose 95% Wilson CIs (Acc) overlap → NOT statistically
    distinguishable from seed variance.
  * The abstract's closing claim "fine histopathological micro-structures exhibit
    acute cross-task interference" is (i) not supported by this ablation (a skip
    ablation is not a cross-task effect) and (ii) conflates the genuine
    TCGA(84) vs PANDA(31) Dice gap, which is DATASET DIFFICULTY (macro lesion vs
    micro-gland), not a skip-connection phenomenon.
- Conclusion: keep the matched comparisons, reframe the finding as a NULL result
  ("bottleneck-only representations are sufficient in this architecture; no
  detectable skip penalty on either dataset; effects within single-run variance"),
  and demote the "paradox" language. If you want to keep a directional claim,
  you must run 3+ seeds of Runs 08/25 and 10/26 and show the effect persists.

### Fix
1. §4.2: add the Acc 95% CIs to both bullets; rewrite the PANDA bullet:
   "…(Run 10: 34.70% Acc / 31.34% Dice vs Run 26: 35.31% / 28.94%; ΔDice = −2.40).
   The differences are small and within single-run variance (Acc 95% CIs overlap:
   [32.67, 36.73] vs [33.27, 37.35]); we therefore interpret the result as no
   detectable penalty for skip removal, i.e., the bottleneck representation
   suffices for fine glandular structures in this architecture."
2. Abstract last sentence → replace "…coarse macro-lesions tolerate bottleneck
   representations, fine histopathological micro-structures exhibit acute
   cross-task interference" with e.g.: "…the v2 optimization package (lower LR +
   dynamic balancing + stain normalization) degrades fine-grained PANDA
   performance, while coarse macro-lesion tasks remain robust to the bottleneck-
   only representation and to skip removal."
3. Optional (recommended if you keep any directional skip claim): re-run
   `--phase v2 --datasets {tcga,panda} --encoders mobilenet_v2 --no-skip-connections
   --seed {43,44,45}` and the matching skip-on seeds; report mean ± SD.
4. Add the UNREPORTED Macenko ablation (Runs 23/24 vs 10/16) to §4.2 — see §4.3 below.

---

## 2. Feedback item 2 — run numbers & the §4.3 GradNorm comparison

### 2a. "(Runs 1, 2 vs 25, 26)" and "rows 25-26 are No-Macenko"
- Explained by provenance (Section 0): in the STALE aggregator table
  (`paper_results_latex_table.txt`) the no-skip rows are positions 1-2 and rows
  25-26 are the No-Macenko rows, with no run-ID column. In the current main.tex
  table, #25/#26 ARE the No-Skips rows, and every run-ID citation in the body
  checks out against the final CSV (01-06 v1, 07-12 v2, 13-16 PanNuke,
  17-22 teardown, 23-26 ablations).
- Action: delete/archive `paper/paper_results_latex_table.txt` (or regenerate it
  with a leading Run-# column), fix `src/aggregate_results.py` to emit run IDs,
  and add a `--enable-gradnorm` flag so a v1+GN run is even possible (it is NOT
  currently — this caused the Run 18 accident, below).

### 2b. THE BIG ONE: Run 18 is not a GradNorm run
Evidence chain (all in this repo):
1. `run_all_experiments.sh` job 18: `python main.py --phase v1 --datasets panda
   --encoders vgg16 --no-macenko` — the comment "(GradNorm ON implicitly)" is
   WRONG: `src/config.py` has `PHASE_CONFIGS["v1"]["use_gradnorm"] = False` and
   `main.py::apply_phase_config` sets `args.use_gradnorm = phase["use_gradnorm"]`
   unconditionally ("GradNorm toggle is phase-controlled"); the only GradNorm
   flags are `--disable-gradnorm` / `--static-weights` — there is no way to turn
   GradNorm ON under v1.
2. `paper_results_matrix.csv` / `_with_ci.csv`: Run 18 = `PANDA-vgg16-no-macenko`,
   Use GradNorm=False, GradNorm Alpha=0.0 — the label is byte-identical to
   Run 03's, i.e., a seed replicate of the v1 5:1 static config (it additionally
   lacks `--compile`, which Runs 03/20 have).
3. The derived LaTeX table likewise shows no GradNorm check, α=0.0.
4. (Verify in GitHub repo: `checkpoints/summary_18_g4_panda_isolate_gn.json` and
   the log line `Phase v1 applied: ... gradnorm=False, alpha=0` in
   `logs/run_18_g4_panda_isolate_gn.log` — deterministic from the code above.)

Consequences:
- §4.3 bullet 1 ("Introducing GradNorm (α=1.5, Run 18) yields 42.54/40.84,
  isolating a modest initial penalty ΔAcc = −2.85%") is INVALID: it compares two
  STATIC 5:1 runs (45.39 vs 42.54). That 2.85-pt gap is seed variance (Acc CIs
  overlap: [43.26, 47.52] vs [40.43, 44.65]). It must be removed or re-run.
- Table 1 row 18 shows GradNorm = \checkmark — WRONG; must be ---.
- §1 item 3 "isolated GradNorm steps" is affected the same way.

### 2c. §4.3 bullet 2 is LR-matched but Macenko-CONFOUNDED
- Run 17 = `--phase v2 --disable-gradnorm --static-weights --no-macenko` →
  1e-4, GN OFF, RAW input (Macenko=False in CSV).
- Run 09 = v2 → 1e-4, GN ON, Macenko ON.
- So "static (43.35) → GradNorm (30.89), ΔAcc = −12.46" conflates GradNorm AND
  Macenko. The CIs are disjoint ([41.23, 45.47] vs [28.92, 32.86]) — the drop is
  real, but its attribution is not clean. The paper hides the raw-input status:
  row 17 has no ◇ marker and Phase is labeled "v2", contradicting the table
  caption's definition of V2 (GN + Macenko).
- Also note: a 1e-4 GN-on + raw-input PANDA run and a 1e-3 GN-on run do NOT
  exist in the matrix, so NEITHER LR has a clean GN on/off pair.

### Fix — add 3 cheap PANDA×VGG16 runs (the 2×2×2 becomes fully decomposable)
```
# Run 27: 1e-4, GN OFF, Macenko ON  (pair with Run 09 isolates GradNorm at 1e-4)
python main.py --phase v2 --datasets panda --encoders vgg16 --disable-gradnorm --static-weights --num-workers 2

# Run 28: 1e-3, GN ON, Macenko ON   (first true 1e-3 GradNorm run; --lr overrides phase default)
python main.py --phase v2 --datasets panda --encoders vgg16 --lr 0.001 --num-workers 2

# Run 29: 1e-3, GN OFF, Macenko ON  (clean 1e-3 anchor for Run 28)
python main.py --phase v2 --datasets panda --encoders vgg16 --lr 0.001 --disable-gradnorm --static-weights --num-workers 2
```
Decomposition after the runs:
- GN effect @1e-4 (Mac ON):  Run 09 vs Run 27
- GN effect @1e-3 (Mac ON):  Run 28 vs Run 29
- Macenko effect @1e-4:      Run 27 vs Run 17 (raw)
- Macenko effect @1e-3:      Run 29 vs Runs 03/18/20 (raw replicates)
(Only cell unrunnable: 1e-3 + GN + raw — impossible under current phase gating;
document it, or add a `--enable-gradnorm` CLI flag.)

Rewrite §4.3 around the 2×2 (LR × GradNorm) with Macenko held ON; add a
transparency note: "Run 18 was intended as the 1e-3 GradNorm arm but, due to
phase-level GradNorm gating, executed as a static 5:1 control; we re-label it
as a seed replicate of Run 20 and exclude it from the GradNorm analysis."
Re-title §4.3 (e.g., "Isolating Learning Rate from Dynamic Gradient Balancing"),
update all "26-run" counts to "28-run" (abstract, §1, §3 heading, §5), and
relabel row 17 (◇, GN ---, "v2-s: static 1e-4, raw") and row 18 (GN ---,
"seed replicate of Run 20").

---

## 3. Feedback item 3 — minor issues & polish

### 3a. Table 1 density / alignment / IDs
Current: 13 columns in `\resizebox{\textwidth}` (font shrinks inconsistently vs
other text). Patch plan:
- Drop the `Phase` column (redundant with LR + GradNorm + group headers).
- Merge `Paper Acc | Paper Dice` into one `Paper (Acc/Dice)` column ("89 / 97")
  → 12 columns total.
- Use `tabularx` at `\footnotesize` (no resizebox): `l` (Config) + numeric
  columns as `>{\,}r@{\,}` or centered `c` with consistent 2-decimal formatting
  (fix CSV artifacts 94.6→94.60, 78.3→78.30, 34.7→34.70, 91.9→91.90).
- Delta columns: always explicit sign, 2 decimals.
- Group headers: `\midrule` + `\multicolumn{12}{l}{...}` with `\addlinespace[2pt]`.
- Marker legend as a footnote: `◇` = raw input (no Macenko); `×` = skips zeroed;
  note that v1 runs are raw by construction (so ◇ only appears on v2 rows).
- Add the 95% Acc CI as a supplementary table (the `_with_ci` CSV is ready-made)
  — this directly strengthens every "single-run" caveat in the paper.

### 3b. Abstract PANDA range
The reviewer's proposed "41.83–46.15% / 28.94–44.34%" is INTERNALLY
INCONSISTENT: 41.83 is the min of the 1e-3 static subset {Runs 03, 04, 18-22},
but 28.94 belongs to Run 26 (v2, no-skip). The min Dice of that 1e-3-static
subset is 38.06 (Run 21). Consistent options:
- ALL 12 PANDA configurations (what the abstract currently says, and the true
  full range): Acc 30.89–46.15, Dice 28.94–44.34.
- V1 naked baselines (Runs 03-04): Acc 45.15–46.15, Dice 43.30–44.34.
- All 1e-3 static controls (Runs 03, 04, 18-22): Acc 41.83–46.15,
  Dice 38.06–44.34.
Recommendation: report the full 12-config range (transparency) AND the v1-naked
pair range (the actual replication headline):
  "…top-1 accuracy spans 30.89–46.15% and Dice 28.94–44.34% across all 12
  PANDA configurations, with the unweighted 1e-3 baselines (Runs 03-04)
  achieving 45.15–46.15% / 43.30–44.34% (vs 87/88% and 98/99% claimed)."
Mirror the same phrasing in §1 bullet 1.

### 3c. §3.1 float interruption
`table* [t]` sits right after the §3.1 protocol paragraph, so it lands at the
top of the next page, away from its (nonexistent-yet) first reference. Fix:
- Move the whole `table*` environment to the TOP OF SECTION 4 (immediately
  after `\section{Discussion & Methodological Anatomy}`, before §4.1, which is
  where it is first cited) — labels then resolve naturally and §3.1 flows
  uninterrupted.
- Add a pointer at the end of §3.1: "All runs are tabulated in Table 1
  (Section 4)."
- If you keep it in §3, use `[!t]` and/or `\clearpage` before §4; but moving it
  is the cleaner fix.

### 3d. GradNorm hyperparameters (explicitly, in §2.3)
Add to the §2.3 paragraph (all values verified against `src/models.py` /
`src/training.py` / `src/config.py`):
- Initial task weights: w_seg(0) = 5.0, w_cls(0) = 1.0 (inherited from the
  static configuration), held as learnable LOG-weights.
- Update frequency: once per minibatch (target recomputed and weights
  re-normalized to Σw = T = 2 after every optimizer step).
- Optimizer for the weights: the SAME Adam optimizer as the backbone (no
  inner loop), at the task LR (1e-4 for v2).
- Gradient-norm parameter set W: ALL trainable parameters of the shared U-Net
  ENCODER (`unet.encoder`) — the current text says "the final convolutional
  block of the shared backbone", which is WRONG and must be fixed.
- Exponent convention: this implementation targets
  G̅_W · [r_i(t)]^{+α} (α = 1.5), whereas the reference GradNorm
  (Chen et al., ICML 2018) uses the inverse convention [r_i(t)]^{−α} with a
  dedicated inner-loop optimizer. State this explicitly: "we implement a
  parameterized variant of GradNorm and keep it fixed across all v2 runs" —
  a careful reviewer will check this, and the current text claims to "implement
  GradNorm [chen2018gradnorm]" without disclosing the deviation.
- Cite chen2018gradnorm (already in references.bib) and macenko2009method.

### 3e. Math notation consistency
Unify to one style (spaces around `=`, chained for equal-zero):
- §1 bullet 2: `($|Y|=|\hat{Y}|=0$)` → `($|Y| = |\hat{Y}| = 0$)`
- §2.2: `$\text{Dice}=1.0$` → `$\text{Dice} = 1.0$`; keep
  `$|Y| = 0 \land |\hat{Y}| = 0$` but use the same spacing convention;
  `$\text{Dice} \equiv 1.0$` is fine.
- Also unify "88.0%" vs "88%" percentage spacing and "Dice" vs "DSC" (paper
  uses Dice consistently — keep; the original paper uses "dice coefficient").

---

## 4. ADDITIONAL issues found in the full audit (beyond the reviewer's list)

1. **§3.1 is factually wrong about the optimizer:** "Adam optimizer (weight
   decay 10⁻⁴), and cosine annealing schedules" — the code has NEITHER
   (`optim.Adam(params, lr=...)` only; no `weight_decay`, no scheduler anywhere
   in `src/`). Fix the sentence to "Adam with a constant per-phase learning rate
   and default (zero) weight decay" — or implement both and re-run (not
   recommended mid-review).
2. **§3.1 dataset naming:** "TCGA Breast / LGG Pathology" — the original paper's
   TCGA-LGG is BRAIN TUMOR MRI (256×256, N = 3,929 — the same N you cite).
   Verify which TCGA subset `data/TCGA` actually contains; if it's LGG-MRI,
   rename to "TCGA-LGG brain tumor (MRI)". (Also note the paper's title says
   "computational pathology" while two of the four datasets are MRI/X-ray —
   mirror the original's "diverse modalities" wording or narrow the title.)
3. **The Macenko ablation (Runs 23/24) is never discussed in the body.** It is a
   clean, matched, SIGNIFICANT finding: PANDA 34.70 → 40.21 Acc (+5.51; CIs
   [32.67,36.73] vs [38.11,42.31] disjoint) and PanNuke 96.68 → 99.36 (+2.68,
   disjoint), Dice roughly flat. Removing Macenko HELPS fine-grained
   classification here (likely chromatin/texture over-smoothing). Add a
   paragraph to §4.2 — it also partially pre-emptively decomposes the §4.3
   confound (Run 17 is the raw counterpart of Run 09).
4. **No figures in the paper at all.** `paper/dice_degeneracy_curve.csv`
   (empty-mask Dice inflation, based on the 44.08 positive-Dice anchor) is a
   ready-made §2.2 figure. Strongly recommended to include.
5. **Stale/contradictory artifacts in the repo** (a reproducibility red flag if
   the reviewer inspects the GitHub repo): `paper_results_latex_table.txt`
   (old ordering), `docs/technical_diagnostic_report.md` (earlier-round numbers
   and run map), the wrong "GradNorm ON implicitly" comment in
   `run_all_experiments.sh`, and the unused `ali2026cancerpathologydl` bib entry
   (the intro uses a bare \url instead of citing it). Archive/supersede/mark.
6. **`run_all_experiments.sh` lacks any way to run v1 + GradNorm.** Add an
   `--enable-gradnorm` flag (override the phase default) so the intended
   1e-3+GradNorm arm is executable — this is what prevented Run 18 from being
   what the authors thought it was.
7. **PanNuke v2 VGG16 volatility:** the earlier round (technical_diagnostic_report)
   recorded 10.97-39.59 Dice ranges for v2 VGG16 PanNuke vs 39.59 in the final
   round — large run-to-run variance on that cell. Not required by the
   feedback, but if you keep the "VGG16 fragile" narrative, report the
   variance instead of a single point.

---

## 5. Exact patch plan (checklist)

### paper/main.tex
- [ ] P1. Abstract: 26→28 experiments; PANDA sentence → full-range + v1-naked-pair
      wording (§3b); reframe closing sentence (remove "acute cross-task
      interference") (§1).
- [ ] P2. §1 item 1: mirror the range fix; item 3: reword "isolated GradNorm
      steps" → "isolated learning-rate / GradNorm / loss-weighting controls".
- [ ] P3. §2.3: fix W definition ("shared encoder" not "final convolutional
      block"); add the explicit hyperparameter block (initial weights 5/1,
      log-parameterization, per-minibatch update, Σw=2, +α variant disclosure,
      Chen et al. citation) (§3d).
- [ ] P4. §3.1: fix TCGA-LGG naming; fix optimizer sentence (no wd / no
      cosine); add note that Group 4 rows are raw-input; extend protocol for
      Runs 27-29; move Table 1 environment to top of §4; add pointer sentence
      (§3c).
- [ ] P5. Table 1: new column layout (drop Phase, merge Paper, 12 cols,
      tabularx \footnotesize, 2-decimal alignment, signed deltas, ◇/× footnote,
      group headers); fix row 17 (add ◇, GN ---, phase "v2-s static, raw");
      fix row 18 (GN ---, relabel "seed replicate of Run 20"); add rows 27-29
      after Run 22 in Group 4.
- [ ] P6. §4.1: keep Run 04 sentence; optionally add the v1-static sub-range.
- [ ] P7. §4.2: rewrite both bullets with CIs and the null-result framing; add
      Macenko-ablation paragraph (Runs 23/24 vs 10/16) (§1, §4.3).
- [ ] P8. §4.3: rewrite as 2×2 (LR × GradNorm, Mac ON) using Runs 09/17/27/28/29
      + raw anchors 03/18/20; include the Run-18 transparency note; remove the
      "−2.85% GradNorm penalty" and the "multiplicative compounding" claim
      (pending new-run results) (§2).
- [ ] P9. §5: 26→28; update recommendations to match the revised findings
      (add "verify GradNorm implementation conventions" / "report per-run CIs").
- [ ] P10. Math: unify `|Y| = |\hat{Y}| = 0` spacing (§3e); optional
      \includegraphics of the Dice-degeneracy curve in §2.2.

### repo
- [ ] R1. Run Runs 27-29 (commands in §2); regenerate CSVs with
      `src/aggregate_results.py`.
- [ ] R2. `src/aggregate_results.py`: emit a leading Run-# column; verify
      `Use GradNorm` comes from the JSON `use_gradnorm` field (it does) and
      that Run 18 records False.
- [ ] R3. `run_all_experiments.sh`: fix the Group-4 comment block (4.2 line);
      add `--enable-gradnorm` support in `main.py::apply_phase_config`
      (CLI override of `args.use_gradnorm`).
- [ ] R4. Archive `paper/paper_results_latex_table.txt` (e.g., to
      `paper/_superseded/`) with a README note; mark
      `docs/technical_diagnostic_report.md` as SUPERSEDED; add the
      `ali2026cancerpathologydl` citation to the intro.
- [ ] R5. Verify in the GitHub repo: `checkpoints/summary_18_g4_panda_isolate_gn.json`
      (`use_gradnorm: false`) and `logs/run_18_g4_panda_isolate_gn.log`
      ("Phase v1 applied: ... gradnorm=False, alpha=0") — attach as
      supplementary evidence for the reviewer response.

## 6. Suggested reviewer-response posture
- Item 1: "Partially valid — the v2-vs-v1 comparison has been replaced by a
  like-for-like (Run 10 vs Run 26) comparison; the honest effect is small and
  within single-run variance, so we have reframed the claim as a null result
  and removed the 'paradox' framing from the abstract."
- Item 2: "Valid and partially stale: run IDs in the current table are
  consistent; however, auditing our pipeline revealed Run 18 (intended
  1e-3-GradNorm) executed as a static control due to phase-level gating, and
  Run 17's raw-input status was not marked. We have re-labeled both, added
  three matched control runs (27-29) that fully isolate LR, GradNorm and
  Macenko, and rewritten §4.3 accordingly."
- Item 3: adopt all (table layout, ranges, float, GradNorm hyperparameters,
  math spacing) as specified.
