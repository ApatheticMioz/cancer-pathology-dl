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
| F-3 | plan | `paper/generate_figures.py` hardcodes result values (PANDA 34.70→40.21, PanNuke 96.68→99.36, floor points). | OPEN → P2 refactor |
| F-4 | plan | Venue port: llncs→elsarticle; splncs04→num-style bib; build-script 10-page gate obsolete. | OPEN → P1 |
| F-5 | plan | Text: L249 (288w, 6 uniform sentences), L238 (93w opener), L252 (3×43-47-68w) — rhythm uniformity; graphics to absorb. | OPEN → P3/P5 |
| F-6 | plan | Template repetition: "fine-grained"×10, "dynamic gradient balancing"×8, "verified null result"×4, "static 5:1…(Runs 03–04)"×3, "strictly matched"×3, "We therefore"×2, abstract 3×"We" openers. | OPEN → P5 |
| F-7 | plan | Register: "collapse"×6 (overstates 16-pt drop), "infeasible", "acute", "appealing paradigm". | OPEN → P5 |
| F-8 | plan | Passive clusters: L220 (3×), §2.3–2.4 GradNorm/Macenko prose. | OPEN → P5 |
| F-9 | plan | Elsevier declarations missing (AI-writing statement, CRediT, Data Availability, highlights, graphical abstract). | OPEN → P6 |
| F-10 | plan | epoch_log.jsonl has no run-id; needs timestamp-window disambiguation for F4 curves. | OPEN → P2 |

## Closure log

- (none yet)
