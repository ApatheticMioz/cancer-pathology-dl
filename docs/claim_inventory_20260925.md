# Tex numeric-claim inventory vs fold campaign (2026-09-25, tex_rederive_v1)



I now have all the evidence needed (tex, `dice_ci_summary.csv`, all 27 `kfold_*.json`, both `paper_results_matrix*.csv`). This is a read-only inventory — the deliverable is this final message; no files are written.

## Numeric-claim inventory — `paper/all_dice_no_slice.tex` vs. fold-campaign evidence

Legend — **Verdict**: MATCH / STALE / AMBIGUOUS-ESTIMATOR / NO-EVIDENCE. "Fold-aware" = per-fold `final_val_*` mean (JSON) for Acc; `across_folds` pooled per-case PE [95% CI] (CSV) for Dice.

### Abstract (L85)
| Line | Claim as written | Evidence source | Verdict | Corrected / fold-aware value |
|---|---|---|---|---|
| 85 | "82%–90%" acc, "97%–99%" Dice (Rhanoui) | external published target | NO-EVIDENCE (external target, not our measurement) | n/a — audited target |
| 85 | "26 experiments across four datasets" | 27 `kfold_*.json` | **STALE** | 27 configs × 5 folds = 135 fold-runs |
| 85 | "77.74% Dice" (all 4 SIIM, all-empty) | CSV `across_folds` SIIM 05/06/11 = 77.72% [77.70–77.74]; 12 = 77.04% [75.68–77.74]; per-fold 77.60–77.75; ρ=1659/2135=77.70% | **MATCH** | 77.04–77.75% (floor stable across folds) |
| 85 | "26.09%–46.06%" PANDA Acc (12 configs) | JSON per-fold means | **STALE** | 28.21%–44.87% (run18 28.21 [22.81–33.48]; run17 44.87 [43.18–46.08]) |
| 85 | "85.35%–85.50%" PANDA Dice (12 configs) | JSON per-fold val-mean 82.77–84.39%; CSV `across_folds` 98.54–98.60% | **AMBIGUOUS-ESTIMATOR** | 85.35–85.50 = per-epoch val-mean macro-over-present-classes (fold 82.77–84.39%); the true all-slices empty-credit floor is 98.54–98.60% |
| 85 | "43.25%–45.01%" (Runs 03–04) | JSON run03 34.51 [27.29–41.13]; run04 39.91 [38.42–43.37] | **STALE** | 34.51% / 39.91% |
| 85 | "85.35%–85.50%" (Runs 03–04 Dice) | JSON 84.39% [83.83–84.84] both | **AMBIGUOUS-ESTIMATOR** | per-epoch val-mean 84.39%; all-slices floor 98.60% |
| 85 | "88.0% and 99.0% claimed" | external target | NO-EVIDENCE (external) | n/a |
| 85 | "+7.03% on PANDA" (Macenko) | JSON run23 41.38 vs run10 37.56 | **STALE** | +3.82 pts |
| 85 | "+6.64% on PanNuke" (Macenko) | JSON run24 99.24 vs run16 96.49 | **STALE** | +2.75 pts (note: dispatch said +6.94; tex actually says +6.64) |

### Introduction (L97–108)
| Line | Claim | Evidence | Verdict | Corrected |
|---|---|---|---|---|
| 97 | "λ_seg=5.0, λ_cls=1.0", "η=10⁻³" | matrix `Seg/Cls Weight`, `LR` | **MATCH** | protocol confirmed |
| 97 | "82%–90%" / "97%–99%" (Rhanoui) | external | NO-EVIDENCE (external) | n/a |
| 101 | "26-run replication study" | 27 kfold JSONs | **STALE** | 27 configs × 5 folds |
| 103 | "88.0% / 99.0%" claimed | external | NO-EVIDENCE (external) | n/a |
| 103 | "26.09%–46.06% Acc / 85.35%–85.50% Dice across all 12 configurations" | JSON | **STALE** (Acc) / **AMBIGUOUS-ESTIMATOR** (Dice) | Acc 28.21–44.87%; Dice per-epoch 82.77–84.39% (all-slices floor 98.54–98.60%) |
| 103 | "43.25%–45.01% / 85.35%–85.50%" (Runs 03–04) | JSON | **STALE** / **AMBIGUOUS-ESTIMATOR** | 34.51%/39.91%; Dice 84.39% |
| 104 | "99% Dice claims" (SIIM) | external | NO-EVIDENCE (external) | n/a |
| 104 | "77.74%" SIIM floor | CSV across_folds | **MATCH** | 77.04–77.75% |
| 105 | "λ_seg:λ_cls ∈ {1:10,1:1,5:1,10:1}" | matrix | **MATCH** | protocol |
| 105 | "43.16% → 26.09% Acc" (GradNorm) | JSON run20 38.18 → run18 28.21 | **STALE** | 38.18% → 28.21% (Δ −9.97) |
| 108 | "all 26 runs" / "26 runs" | 27 kfold JSONs | **STALE** | 27 configs × 5 folds |

### Figures / pipeline (L128, 137, 146, 152)
| Line | Claim | Evidence | Verdict | Corrected |
|---|---|---|---|---|
| 128 | "26-run factorial … under GroupKFold" | 27 kfold JSONs | **STALE** | 27 configs × 5 folds |
| 137 | "All 26 configurations … 77.74% on SIIM" | 27 kfold JSONs; CSV | **STALE** (count) / **MATCH** (77.74) | 27 configs; 77.04–77.75% |
| 146 | "lesion-free fraction approaches 78%" | 1659/2135=77.70% | **MATCH** | 77.7% |
| 152 | "97–99% Dice alongside 82–90% accuracy" | external | NO-EVIDENCE (external) | n/a |

### Method / metrics (L157, 180, 195, 200, 211, 215, 219, 226)
| Line | Claim | Evidence | Verdict | Corrected |
|---|---|---|---|---|
| 157 | "hidden width 256, dropout 0.5" | not in fold JSONs | NO-EVIDENCE (code detail) | n/a |
| 180 | "ε = 10⁻⁸" | dice convention | **MATCH** | protocol |
| 195 | "ρ = 1,659/2,135 = 0.777" | CSV n=2135, neg=1659 | **MATCH** | 0.777 |
| 195 | "77.74% (Runs 05,06,11,12) is the all-empty floor" | CSV across_folds | **MATCH** | 77.04–77.75% |
| 195 | "analytic macro-Dice floor of 85.35% across all 12 PANDA runs … despite zero true foreground overlap" | JSON per-fold 82.77–84.39%; CSV across_folds 98.54–98.60% | **AMBIGUOUS-ESTIMATOR** | 85.35% = per-epoch val-mean macro-over-present-classes; the all-slices empty-credit floor (the actual empty-mask artifact) is 98.54–98.60%, not 85.35% |
| 200 | "ρ=77.7%"; "77.74% would report 95.04%"; "all-empty floor of 77.74%" | CSV | **MATCH** | 77.04–77.75% |
| 211 | "w_seg(0)=5.0, w_cls(0)=1.0" | matrix | **MATCH** | protocol |
| 215 | "α = 1.5" | matrix `GradNorm Alpha` | **MATCH** | protocol |
| 219 | "dedicated optimizer (learning rate 0.025)" | canonical probe config | **MATCH** | protocol |
| 226 | "I₀=255"; "0.15 threshold"; "1st–99th percentile" | Macenko params | **MATCH** | protocol |

### Table 1 + dotplot (L243–262)
| Line | Claim | Evidence | Verdict | Corrected |
|---|---|---|---|---|
| 243 | "26-run … (single seed 42, single split — not seed variance)" | 27 kfold JSONs w/ fold_sd | **STALE** (count) + **FALSE disclaimer** | 27 configs × 5 folds; fold CIs now exist |
| 243 | "SIIM 77.74% / PANDA 85.35% … degeneracy floors" | CSV | **MATCH** (SIIM) / **AMBIGUOUS-ESTIMATOR** (PANDA) | PANDA all-slices floor = 98.54–98.60% |
| 249 | TCGA "89.07–95.12" Acc, "75.24–84.47" Dice | JSON per-fold means | **STALE** | Acc 92.86–95.62%; Dice (per-fold) 85.64–89.22% / across_folds 85.33–88.85% |
| 250 | PANDA "26.09–46.06" Acc, "85.35–85.50" Dice | JSON/CSV | **STALE** / **AMBIGUOUS-ESTIMATOR** | Acc 28.21–44.87%; Dice per-epoch 82.77–84.39% (all-slices 98.54–98.60%) |
| 251 | SIIM "78.22–83.65" Acc, "77.74" Dice | JSON/CSV | **STALE** (Acc) / **MATCH** (Dice) | Acc 77.71–82.63%; Dice 77.04–77.75% |
| 252 | PanNuke "91.65–99.18" Acc, "40.65–69.78" Dice | JSON/CSV | **STALE** | Acc 78.70–99.24%; Dice per-fold 59.77–70.76% / across_folds 68.44–77.61% |
| 260 | "26-run"; "26.09%–46.06%"; "85.35%–85.50%"; "77.74%" | JSON/CSV | **STALE** / **AMBIGUOUS-ESTIMATOR** / **MATCH** | as above |

### Qualitative fig + datasets (L267–277)
| Line | Claim | Evidence | Verdict | Corrected |
|---|---|---|---|---|
| 267 | "single seed 42, single group-aware split" | 27 kfold JSONs | **FALSE disclaimer** | 5-fold GroupKFold per config; fold stats exist |
| 273 | TCGA "N=3,929; 110 patients" | JSON `samples=3929` | **MATCH** (N) / NO-EVIDENCE (110 patients, literature) | N=3,929 confirmed |
| 274 | PANDA "N=10,616 tiles; biopsy-level GroupKFold" | JSON `samples=10516` | **STALE** (N) / **MATCH** (isolation) | N=10,516 |
| 275 | SIIM "N=12,047 images" | JSON `samples=10675` | **STALE** | N=10,675 |
| 276 | PanNuke "N=7,901 patches" | JSON `samples=7901` | **MATCH** | N=7,901 |

### Training protocol (L279)
| Line | Claim | Evidence | Verdict | Corrected |
|---|---|---|---|---|
| 279 | "up to 50 epochs … patience 10" | JSON `last_completed_epoch`/early-stop | **MATCH** | protocol |
| 279 | "batch size 32" | JSON `batch_size=32` | **MATCH** | 32 |
| 279 | "fixed seed 42" | protocol | **MATCH** | (now 5 folds, not 1 seed) |
| 279 | "Adam … constant per-phase LR … zero weight decay" | JSON | **MATCH** | protocol |
| 279 | "global gradient-norm clipping (max‖∇‖=1.0) … [±log 10] … stabilizers only and do not alter the reported metrics" | JSON `grad_clip_max_norm=1.0` | **MATCH** (value) but **disclaimer must flip to claim** | clip=1.0 confirmed in all fold JSONs; state as a protocol element, not a non-impactful aside |
| 279 | "η=10⁻³ (v1)"; "α=1.5, η=10⁻⁴ (v2)" | matrix | **MATCH** | protocol |
| 279 | *(bf16 / mixed precision)* | **not present in tex**; not in fold JSONs | NO-EVIDENCE / not claimed | tex makes no bf16 claim |

### PANDA anatomy (L284–291)
| Line | Claim | Evidence | Verdict | Corrected |
|---|---|---|---|---|
| 284 | "88.0% / 99.0%" claimed | external | NO-EVIDENCE (external) | n/a |
| 284 | "26.09%–46.06% / 85.35%–85.50%" | JSON/CSV | **STALE** / **AMBIGUOUS-ESTIMATOR** | Acc 28.21–44.87%; Dice per-epoch 82.77–84.39% |
| 284 | "43.25% / 85.50% (VGG16)" (run03) | JSON 34.51% / 84.39% | **STALE** / **AMBIGUOUS-ESTIMATOR** | 34.51% / 84.39% |
| 284 | "45.01% / 85.35% (MobileNetV2)" (run04) | JSON 39.91% / 84.39% | **STALE** / **AMBIGUOUS-ESTIMATOR** | 39.91% / 84.39% |
| 284 | "Run 04 is the strongest single configuration" | JSON: run17 44.87% > run04 39.91% | **STALE** (ranking) | Run 17 (44.87%) is strongest fold-mean |
| 286 | "1,010 teams"; "κ=0.862 (0.840–0.884)"; "0.868 (0.835–0.900)"; "κ=0.68"; "0.435" | literature (bulten/allsbrook) | NO-EVIDENCE (in fold results) | external citations |
| 286 | "88.0% accuracy on 128×128" | external target | NO-EVIDENCE (external) | n/a |
| 291 | "Run 03 43.25%"; "Run 10 33.70%"; "Run 18 26.09%" | JSON 34.51 / 37.56 / 28.21 | **STALE** | 34.51% / 37.56% / 28.21% |
| 291 | "canonical … plateaus at 41.16% over 15 epochs, 46.84 pts below 88.0%" | JSON canonical per-fold 22.70% [11.55–35.81] | **STALE** | 22.70% [11.55–35.81] (Δ vs 88% ≈ −65.3) |

### Skip ablation (L298–301)
| Line | Claim | Evidence | Verdict | Corrected |
|---|---|---|---|---|
| 298 | TCGA "93.32% vs 94.22% (Δ −0.90)"; "80.97% vs 83.85% (Δ −2.88)" | JSON run25 95.28 [93.51–97.02] vs run08 95.26 [93.51–97.16]; Dice 86.16 vs 89.22 | **STALE** | Acc Δ +0.02; Dice Δ −3.06 (still null) |
| 299 | PANDA "36.88% / 85.35% vs 33.70% / 85.35% (Δ +3.18, 0.00)" | JSON run26 37.44 [36.09–38.18] vs run10 37.56 [37.09–38.18]; Dice 84.29 vs 82.77 | **STALE** / **AMBIGUOUS-ESTIMATOR** | Acc Δ −0.12; Dice Δ +1.52 (still null) |
| 301 | "95% Wilson CIs … [91.34,94.87] vs [92.35,95.65]; [34.85,38.97] vs [31.71,35.75]"; "would require multi-seed replication" | fold CIs exist | **STALE** (CIs) + **FALSE disclaimer** | replace single-split Wilson with fold CIs; multi-fold already done |

### Macenko ablation (L304)
| Line | Claim | Evidence | Verdict | Corrected |
|---|---|---|---|---|
| 304 | "PANDA +7.03 (40.73% vs 33.70%)" | JSON run23 41.38 vs run10 37.56 | **STALE** | +3.82 (41.38% vs 37.56%) |
| 304 | "PanNuke +6.64 (99.18% vs 92.54%)" | JSON run24 99.24 vs run16 96.49 | **STALE** | +2.75 (99.24% vs 96.49%) |
| 304 | "disjoint 95% CIs [38.65,42.85] vs [31.71,35.75]; [98.59,99.51] vs [91.13,93.73]" | single-split Wilson | **STALE** | fold CIs overlap more (resolvability weakens) |
| 304 | "Dice flat on PANDA (85.35%)" | JSON 84.39% | **AMBIGUOUS-ESTIMATOR** | per-epoch 84.39%; all-slices 98.60% |
| 304 | "rises +11.17 on PanNuke (69.78% vs 58.61%)" | JSON run24 70.76 vs run16 62.25 | **STALE** | +8.51 (70.76% vs 62.25%) |

### GradNorm / LR / λ-sweep (L309–327)
| Line | Claim | Evidence | Verdict | Corrected |
|---|---|---|---|---|
| 309 | "Run 18 26.09% vs Run 20 43.16%"; "Run 23 40.73% vs Run 10 33.70%"; "Run 24 99.18% vs Run 16 92.54%"; "Runs 17 vs 20: 46.06% vs 43.16%"; "Runs 25 vs 08: 93.32% vs 94.22%"; "Runs 26 vs 10: 36.88% vs 33.70%" | JSON per-fold means | **STALE** (all six pairs) | 28.21/38.18; 41.38/37.56; 99.24/96.49; 44.87/38.18; 95.28/95.26; 37.44/37.56 |
| 317 | "43.16% → 26.09% (CI [24.26,28.01] vs [41.05,45.28], Δ −17.07%); Dice flat 85.35%" | JSON run20 38.18 → run18 28.21 | **STALE** / **AMBIGUOUS-ESTIMATOR** | 38.18% → 28.21% (Δ −9.97); Dice per-epoch 84.39% |
| 319 | "26.09% … Run 09 33.03% … ≈26%–33% floor" | JSON run18 28.21 [22.81–33.48]; run09 33.34 | **STALE** | 28.21% / 33.34%; floor ≈22.8–33.4% |
| 322 | "Run 20 43.16% → Run 17 46.06%"; "Runs 19–22 43.11%–43.87%"; "Run 18 26.09% vs 43.16%"; "canonical 41.16%" | JSON | **STALE** | 38.18%→44.87%; 36.51–39.71%; 28.21% vs 38.18%; canonical 22.70% |
| 327 | "1:10 43.68%; 1:1 43.11%; 5:1 43.16%; 10:1 43.87%"; "Dice 85.35%,85.35%,85.35%,85.47%"; "43.11%–43.87%" | JSON run21 36.51 / run19 36.82 / run20 38.18 / run22 39.71 | **STALE** / **AMBIGUOUS-ESTIMATOR** | Acc 36.51–39.71%; Dice per-epoch 84.39% |

### Conclusion (L332)
| Line | Claim | Evidence | Verdict | Corrected |
|---|---|---|---|---|
| 332 | "Across 26 experiments" | 27 kfold JSONs | **STALE** | 27 configs × 5 folds |
| 332 | "88%–99%" claimed | external | NO-EVIDENCE (external) | n/a |
| 332 | "43.16% → 26.09% Acc" | JSON 38.18% → 28.21% | **STALE** | 38.18% → 28.21% |
| 332 | "95% Wilson score intervals on matched single-run ablation comparisons" | fold CIs exist | **FALSE disclaimer** | use fold-level CIs (across_folds), not single-run Wilson |

### METHODOLOGY sentences that must flip (disclaimer → claim)
| Line | Sentence | Required flip |
|---|---|---|
| 85 / 103 / 137 / 148 / 332 | "strict patient/biopsy-level isolation (GroupKFold)" (blanket) | **Per-dataset**: PANDA = image/biopsy-level GroupKFold; SIIM = trivial stratified (1 img/patient → GroupKFold degenerate); TCGA = patient-level GroupKFold; PanNuke = image-level (no patient grouping) |
| 274 | "PANDA … biopsy-level GroupKFold isolation" | keep (correct for PANDA); add the other three datasets' isolation levels |
| 279 | clip disclosure "…stabilizers only and do not alter the reported metrics" | flip to a **claim**: max‖∇‖=1.0 + log-weight cap [±log10] are confirmed protocol elements (present in all 27 fold JSONs), not a non-impactful aside |
| 243 / 267 / 301 / 332 | "single seed 42, single split" / "single group-aware split" / "would require multi-seed replication" / "single-run ablation comparisons" | all **FALSE** now — 5-fold GroupKFold stats (PE + 95% CI + fold_sd) exist for all 27 configs; replace with fold-aware CIs |

---

### 5-line summary of the biggest stale clusters
1. **PANDA accuracy range** is the largest stale cluster: single-split "26.09%–46.06%" → fold-aware **28.21%–44.87%** (run18 26.09→28.21, run17 46.06→44.87); every individual PANDA run value (03,04,09,10,17–23,26) shifts, and "Run 04 is strongest" is now false (Run 17 leads at 44.87%).
2. **The "85.35% PANDA Dice floor" is mislabeled (AMBIGUOUS-ESTIMATOR)**: 85.35–85.50% is the per-epoch val-mean macro-over-present-classes (fold 82.77–84.39%); the true all-slices empty-credit floor — the actual empty-mask artifact — is **98.54–98.60%**, so the tex's "floor … despite zero foreground overlap" framing names the wrong estimator.
3. **Macenko ablation deltas shrink materially**: PANDA +7.03→**+3.82**, PanNuke +6.64→**+2.75** (and PanNuke Dice +11.17→+8.51), so the "only statistically resolvable shift" claim weakens under fold CIs.
4. **GradNorm-collapse numbers all move**: 43.16→26.09 becomes **38.18→28.21** (Δ −17.07→−9.97), and the canonical decoupled probe "41.16% plateau" → **22.70% [11.55–35.81]**.
5. **Counts/sizes/disclaimers**: "26 experiments" → **27 configs × 5 folds**; dataset sizes PANDA 10,616→**10,516** and SIIM 12,047→**10,675** (TCGA 3,929 and PanNuke 7,901 match); and every "single split / single seed / needs multi-seed" disclaimer is now **false** since fold PE+CI+fold_sd exist for all 27 configs.