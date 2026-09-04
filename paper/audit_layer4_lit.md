# Layer 4 Audit: Citation Integrity — `paper/references.bib` vs `paper/main.tex`

Auditor: Lead Scientific Architect (Plan Mode). Scope: key-level closure between bib and tex,
entry metadata integrity, and load-bearing literature claims requiring external verification.
VERIFY-LIT items are gated web-verification steps for Phase 2 (WebSearch) with fallbacks.

---

## 1. Key-level closure

**Cited in tex (14 keys, all resolve in bib):** litjens2017survey (L63), campanella2019clinical
(L63), lu2024visual (L63), lu2024multimodal (L63), vorontsov2024virchow (L63), ronneberger2015u
(L63), rhanoui2025multitask (L65, L89, L117, L163, L211), simonyan2014very (L65),
sandler2018mobilenetv2 (L65), chen2018gradnorm (L71, L127), ali2026cancerpathologydl (L74),
reinke2024metrics (L104), macenko2009method (L144, L148, L222), bulten2022artificial (L229).

**In bib but never cited (2):**
- `gamper2020pannuke` — PanNuke is USED (L219) but the entry is never cited. **Required fix:
  cite at first dataset mention (L219).**
- `vahadane2016structure` — never cited. Either cite at L148 as the alternative
  structure-preserving stain-normalization method (one clause) or remove the entry.

**Mandatory new citations (datasets/methods used but uncited):**
- TCGA-LGG 2D MRI release (L216) — no bib entry. VERIFY-LIT the canonical source for the exact
  2D slice release used (candidate: Buda et al., Comput. Biol. Med. 2019; or TCGA-LGG
  collection paper) and add.
- SIIM-ACR pneumothorax dataset (L218) — no bib entry. Add @misc for the RSNA/SIIM-ACR Kaggle
  challenge (2019) with URL.
- PCGrad (L251) — named without citation. Add Yu et al., NeurIPS 2020 ("Gradient Surgery for
  Multi-Task Learning") or delete the mention (the sentence is being rewritten anyway — audit
  item C4).
- PANDA dataset at first mention (L217): `bulten2022artificial` exists and is cited only at
  L229; add the cite at L217 for dataset provenance.

Net bib delta: +2 mandatory (TCGA-LGG, SIIM-ACR), +1 optional (PCGrad) or −1 (vahadane) —
budgeted within the zero-net-expansion rule via prose tightening.

## 2. Entry metadata audit

| Key | Verdict | Notes |
|---|---|---|
| rhanoui2025multitask | VERIFY (repo-internal, cheap) | MDPI "Onco" 5(3):34, doi:10.3390/onco5030034 — consistent with the local source copy `paper/…onco-05-00034-v3.md`; verify title/authors verbatim against that file. |
| bulten2022artificial | VERIFY-LIT (title) | Nat Med 28(1):154–163 plausible; confirm exact article title (variants circulate with/without a subtitle) and the participants count (tex L229 ">1,000 teams"). |
| gamper2020pannuke | OK (metadata) | CMIG 80:101703 (2020). Uncited — see §1. |
| chen2018gradnorm | OK + VERIFY-LIT (claims) | ICML/PMLR 794–803 ✓. The manuscript leans on canonical specifics (separate weight optimizer; Adam lr=0.025 for weights; stop-gradient; α=1.5 recommended) — confirm each appears in the paper before §2.3/§4.3 assert it. |
| reinke2024metrics | NAMING MISMATCH | Bib = "Understanding metric-related pitfalls in image analysis: a guide for biomedical researchers" (Nat Methods 21(2):182–194). Tex L104 calls it "Metrics Reloaded" — that name belongs to the sibling recommendations paper (Reinke et al., Briefings in Bioinformatics 2024). Fix the tex name, or cite both. VERIFY-LIT both titles/venues. |
| macenko2009method | OK + VERIFY-LIT | ISBI 2009, 1107–1110. Confirm the original OD definition is base-10 (`log10(I0/I)`) as asserted at L148. |
| vahadane2016structure | AUTHOR BUG (moot if removed) | "Zheng, Nassir" should be "Zheng, Yanning" (TMI 35(8):1962–1971, 2016 — rest OK). |
| ronneberger2015u | OK | MICCAI 234–241. |
| simonyan2014very | OK | arXiv:1409.1556. |
| sandler2018mobilenetv2 | OK | CVPR 4510–4520. |
| litjens2017survey | OK | MedIA 42:60–88. |
| campanella2019clinical | OK | Nat Med 25(8):1301–1309. |
| ali2026cancerpathologydl | OK | @misc; URL matches the repository; year 2026. |
| lu2024visual | **HIGH SUSPICION — metadata corruption** | Entry: title "A visual foundation model for cancer pathology" with first author "Lu, Ming Y". UNI's Nat Med 30:863–874 paper is Chen, Richard J. et al. (title uses "vision foundation model"); Lu, Ming Y. is first author of the CONCH paper. Author/title appear conflated. VERIFY-LIT and correct authors+title+DOI. |
| lu2024multimodal | **HIGH SUSPICION — metadata corruption** | Title "A multimodal foundation model for computational pathology"; CONCH's actual title is "A visual-language foundation model for computational pathology" (Nat Med 30:853–862, Lu et al.). VERIFY-LIT and correct. |
| vorontsov2024virchow | VERIFY-LIT | "A foundation model for clinical-grade computational pathology" Nat Med 30(3):875–886 — title/venue plausible; verify author list (the "…Khalvati, Farzad and others" tail is uncertain). |

## 3. Load-bearing literature claims in tex requiring verification (with fallbacks)

1. **L229 κ 0.86–0.90 (Bulten)** — VERIFY-LIT against the paper's reported external-validation
   quadratic weighted κ. Fallback: quote the exact verified range with page/figure reference.
2. **L229 "corresponding to tile-level top-1 accuracies 40%–52%"** — not a Bulten result; our
   inference. Fallback (preferred): delete the range or restate as an explicit derivation with
   stated assumptions (class distribution, cost matrix).
3. **L229 inter-observer κ "0.65–0.75"** — uncited. VERIFY-LIT Gleason inter-observer studies
   (candidates: Allsbrook et al. 2001; Özdamar et al.; Egevad et al.) and use the verified
   range for the cited populations (general vs. urologic pathologists report different κ).
4. **§2.3/L127+ canonical GradNorm specifics** — VERIFY-LIT α=1.5 default, separate optimizer,
   Adam lr=0.025 for weights, stop-gradient in Chen et al. 2018.
5. **L148 canonical Macenko base-10 OD** — VERIFY-LIT.
6. **L104 empty-mask conventions attributed to Reinke et al.** — VERIFY-LIT that the cited
   pitfalls guide discusses empty-reference-mask handling; else soften attribution.
7. **Foundation-model framing sentence (L63)** — after correcting the two corrupted entries,
   confirm venue/year triples (all three claimed Nat Med 30(3) 2024).

## 4. Style/consistency notes

- Bibliography style `splncs04` ✓ (LNCS). Protect title capitalization with braces where needed
  (repo convention from commit 41cd5dc: "protect bibtex note capitalization").
- DOI fields present for most entries; keep consistent (add DOIs for the two new dataset
  entries where applicable; Kaggle SIIM entry uses URL).
- `\cite` vs LNCS numeric style: fine as-is.
