# AUDIT MANIFEST — Q1 Revision Campaign (CBM)

Branch: `paper/q1-revision` · Baseline tag: `lncs-camera-ready-2026-09` (commit `6a16488`)
Target venue: Computers in Biology and Medicine (Elsevier, JCR Q1)
This manifest tracks every finding opened during the revision. Closure gate: 100% of IDs
must be `[RESOLVED: commit_sha / verified_slice]`, `[DEFERRED: issue]`, or `[WONTFIX: rationale]`
before each merge to `main` and before the submission tag.

## Invariants (carried from REVISION_MASTER_PLAN.md)

- (I1) Every number in tex/figures/captions traceable to a repo artifact (CSV/JSON/log). Automated diff gate.
- (I2) Accuracy CIs only — Dice CIs do not exist in the repo; never imply them.
- (I3) Single-seed (42), single-split framing preserved; no multi-seed or per-fold overclaims.
- (I4) Figures: zero hardcoded result values; all read from CSV/JSON at build time.
- (I5) Humanization removes filler/drama only; scientific claims byte-stable.

## Findings

| ID | Phase | Finding | Status |
|---|---|---|---|
| F-1 | plan | Dice CIs unavailable (no per-slice Dice stored anywhere). | WONTFIX: per-slice dumps were never logged; retraining out of scope. Figures/text show Acc CIs only (I2). |
| F-2 | plan | No multi-seed / per-fold variance data (seed=42, k_folds=None everywhere). | WONTFIX: framing constraint, documented in prose (I3). |
| F-3 | plan | `paper/generate_figures.py` hardcodes result values (PANDA 34.70→40.21, PanNuke 96.68→99.36, floor points). | RESOLVED: bcb5f0a (figures ported onto `paper/figures/` style/loaders package; all values read CSV/JSON at build time, zero hardcoded results; collision fixes verified 4f1d7e1) |
| F-4 | plan | Venue port: llncs→elsarticle; splncs04→num-style bib; build-script 10-page gate obsolete. | RESOLVED: e19b372 (14pp, body byte-stable, 0 llncs-isms; *.spl gitignore housekeeping deferred) |
| F-5 | plan | Text: L249 (288w, 6 uniform sentences), L238 (93w opener), L252 (3×43-47-68w) — rhythm uniformity; graphics to absorb. | RESOLVED: 9a36f75 + e3942c2 + b62c1b4 (evidence absorbed into Figures 4/5/6) and dee7780 (numeric prose deleted from text, carried by captions); remaining prose rhythm rewritten ae25901 |
| F-6 | plan | Template repetition: "fine-grained"×10, "dynamic gradient balancing"×8, "verified null result"×4, "static 5:1…(Runs 03–04)"×3, "strictly matched"×3, "We therefore"×2, abstract 3×"We" openers. | RESOLVED: ae25901 (de-templating rewrite; numeric-token multiset diff 546/546 identical before/after) |
| F-7 | plan | Register: "collapse"×6 (overstates 16-pt drop), "infeasible", "acute", "appealing paradigm". | RESOLVED: ae25901 (register de-escalation; §4.1 title retained as the single deliberate use) |
| F-8 | plan | Passive clusters: L220 (3×), §2.3–2.4 GradNorm/Macenko prose. | RESOLVED: ae25901 (active-voice rewrite where convention permits) |
| F-9 | plan | Elsevier declarations missing (AI-writing statement, CRediT, Data Availability, highlights, graphical abstract). | RESOLVED: dbe083b (declarations block before bibliography, highlights.tex 5×≤85 chars, graphical_abstract.pdf, cover_letter.md) |
| F-10 | plan | epoch_log.jsonl has no run-id; needs timestamp-window disambiguation for F4 curves. | RESOLVED: 54d814c (run_epoch_windows loader + CSV assertion; 17/26 runs attributable, 9 SKIPPED — concurrent same-(dataset,encoder) interleaving. F4 curves limited to attributable runs; endpoints for all 26 live in F3/F5/F6. Caption must state this.) |
| F-11 | p2 | Two vLLM engine crashes during P2 dispatch (idle timeout; EngineCore fault, 3361s). Mitigation: slices split smaller; task state verified via git before re-dispatch. | RESOLVED: process note (no artifact impact — crashed dispatch wrote no files) |
| F-12 | p7 | Table 1 + Figures 4–7 end-flushed to float pages after the bibliography: `[t]`-only placement on 55–75%-height floats under class default fractions (topfraction 0.7) queues them past the references. | RESOLVED: see closure log (float fractions relaxed + `[!tbp]` on the five large floats; vision-QA pp. 8/9/11/13/15) |

## Closure log

- 2026-09-13 (P7b, this commit): F-3, F-5…F-9 closed against their phase commits (above). F-12 opened and closed in the same commit: preamble float-fraction relaxation (`topfraction 0.9`, `floatpagefraction 0.7`, `totalnumber 4`) + `[!tbp]` on Table 1 and Figures 4–7. Result: 18pp, 0 overfull, 0 undefined; floats interleaved in order — Table 1 p.8, Fig 4 p.9, Fig 5 p.11, Fig 6 p.13, Fig 7 p.15 (shared with declarations block), References p.16–18. Page-by-page vision QA passed on all 18 pages (7 figures, 1 table, declarations, footer).
- 2026-09-13 (P7a): overfull elimination, commit 3982a19 — `\emergencystretch=3em` + clean `ps@pprintTitle` redefinition; 9→0 overfull hboxes.
