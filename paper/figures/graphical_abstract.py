"""Graphical Abstract — wide CBM banner (Claimed-vs-Measured / Why / Audit).

Replaces the old square (489×521 pt) graphical abstract with a ~2.5:1 wide
banner that is readable at the CBM 5×13 cm (h×w) / 96 dpi requirement
(minimum 531×1328 px h×w). The canvas is ``figsize ≈ (13.2, 5.3) cm`` so the
300 dpi raster comfortably exceeds the minimum pixel budget.

Three-panel left→right story (minimal text, standalone, no captions):

* (a) **Claimed vs measured** — a compact dot strip: the published claimed
  accuracy (82–90 %) as a black dashed band / open diamonds vs. the
  per-dataset *measured* accuracy ranges (dataset colors; e.g. PANDA
  29.04–46.15). PanNuke has no published target, so it carries no claimed
  marker.
* (b) **Why** — three separate off-axes rows (nested gridspec, equal
  height), each holding one cause: (i) empty-mask Dice crediting (chip
  glyph + single-line label + single-line 77.74 % floor value), (ii)
  patient-level leakage (GroupKFold glyph + label + value), (iii) task
  interference (arrow glyph + label + 45.39 % → 29.04 % value).
* (c) **Audit protocol** — three checklist glyphs (group-aware splitting,
  explicit Dice evaluation convention, 95 % Wilson intervals) plus a
  two-line release note and the repo URL (small, bottom right, wrapped to
  two lines so it fits with a ≥ 0.02 right margin).

Data contract (I4 — no hard-coded result values)
------------------------------------------------
Every number is read from the ground-truth CSVs via the existing loaders:

* per-dataset measured/claimed accuracy ranges →
  :func:`paper.figures.loaders.results_matrix`
  (``paper/paper_results_matrix_with_ci.csv``);
* the SIIM empty-over-empty Dice floor (77.74 %) → the SIIM ``Macro Dice``
  column of the same matrix (all four SIIM runs are all-empty, so this is
  the floor);
* the task-interference arrow (45.39 % → 29.04 %) → CSV rows 20 and 18
  (the static 5:1 baseline and the isolated GradNorm arm on PANDA×VGG16).

The only non-data constant is the repository URL (a provenance string, not a
result value).

Output
------
``paper/graphical_abstract.pdf`` (vector) + ``paper/graphical_abstract.png``
(300 dpi, for QA). The PDF page is saved with ``bbox_inches=None`` (the full
figure size) so the page aspect is the locked ~2.5:1 banner; this is the one
deliberate deviation from :func:`paper.figures.style.save_figure` (which
forces ``bbox_inches="tight"`` and would shrink the page to the content
bbox, breaking the aspect contract).

``verify()`` asserts: the PDF page aspect is within 5 % of 2.5:1 (and the
cm size is within 13×5.2 to 13.4×5.4 cm); the three panel titles do not
overlap; the PANDA measured range printed in panel (a) matches the CSV;
every Text artist's window extent is inside the figure bbox; and no two
Text artists in the same axes overlap (1 px tolerance).
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

from . import style
from .loaders import REPO_ROOT, results_matrix

# Apply the locked style contract once for this module.
style.apply()

# ---------------------------------------------------------------------------
# Locked banner geometry (CBM 5×13 cm h×w, ~2.5:1 wide).
# ---------------------------------------------------------------------------

#: Banner size in centimetres (width, height). Printed at ~13 cm wide.
_BANNER_CM = (13.2, 5.3)
_CM_PER_IN = 2.54

#: Repository URL (provenance string — the only non-data constant).
_REPO_URL = "github.com/ApatheticMioz/cancer-pathology-dl"

#: Colorblind-safe accent colors (Okabe-Ito family, matching style.DATASET_COLORS).
_OKABE_ORANGE = "#E69F00"
_OKABE_SKY = "#56B4E9"
_OKABE_GREEN = "#009E73"
_OKABE_VERM = "#D55E00"
_OKABE_PURPLE = "#CC79A7"
_NEUTRAL_GRAY = "#6E6E6E"
_LIGHT_GRAY = "#E8E8E8"

#: Minimum font size (pt) so every element prints ≥7 pt at 13 cm width.
_MIN_FONT = 7.5


# ---------------------------------------------------------------------------
# Data access (I4 — every value from the CSV, never hard-coded)
# ---------------------------------------------------------------------------

def _dataset_acc_ranges(matrix: pd.DataFrame) -> list[dict]:
    """Per-dataset measured + claimed accuracy ranges, in display order.

    Returns a list of dicts with keys ``name``, ``color``, ``meas_lo``,
    ``meas_hi``, ``claimed_lo`` / ``claimed_hi`` (None when no published
    target, i.e. PanNuke).
    """
    order = [
        ("TCGA", "TCGA-LGG", style.DATASET_COLORS["TCGA"]),
        ("PANDA", "PANDA", style.DATASET_COLORS["PANDA"]),
        ("SIIM", "SIIM-ACR", style.DATASET_COLORS["SIIM"]),
        ("PANNUKE", "PanNuke", style.DATASET_COLORS["PANNUKE"]),
    ]
    out = []
    for key, name, color in order:
        g = matrix[matrix["Dataset"] == key]
        meas = g["Accuracy (%)"]
        claimed = g["Paper Acc (%)"]
        has_claimed = bool(claimed.notna().any())
        out.append({
            "name": name,
            "color": color,
            "meas_lo": float(meas.min()),
            "meas_hi": float(meas.max()),
            "claimed_lo": float(claimed.min()) if has_claimed else None,
            "claimed_hi": float(claimed.max()) if has_claimed else None,
        })
    return out


def _siim_empty_floor(matrix: pd.DataFrame) -> float:
    """The SIIM empty-over-empty Dice floor (all four SIIM runs are all-empty,
    so the mean of their Macro Dice is the floor)."""
    return float(matrix[matrix["Dataset"] == "SIIM"]["Macro Dice (%)"].mean())


def _task_interference(matrix: pd.DataFrame) -> tuple[float, float]:
    """(static 5:1 baseline acc, isolated GradNorm acc) on PANDA×VGG16.

    The CSV is in run order: row 20 (idx 19) is the static 5:1 baseline
    (Run 20) and row 18 (idx 17) is the isolated GradNorm arm (Run 18).
    Both are disambiguated by their Run Label + feature set so the values
    cannot silently desync from the data.
    """
    static = matrix[
        (matrix["Run Label"] == "PANDA-vgg16-no-macenko")
        & (matrix["Use GradNorm"] == False)  # noqa: E712
        & (matrix["LR"] == 0.001)
        & (matrix["Lambda Ratio (Seg:Cls)"] == "5:1")
    ]
    # Two rows share that label (Run 03 Group-1 baseline and Run 20 Group-4
    # isolation); the CSV is in run order, so the *last* such row is Run 20.
    static_acc = float(static["Accuracy (%)"].iloc[-1])

    grad = matrix[
        (matrix["Run Label"] == "PANDA-vgg16-no-macenko-alpha=1.5")
        & (matrix["Use GradNorm"] == True)  # noqa: E712
        & (matrix["LR"] == 0.001)
        & (matrix["Lambda Ratio (Seg:Cls)"] == "5:1")
    ]
    grad_acc = float(grad["Accuracy (%)"].iloc[0])
    return static_acc, grad_acc


# ---------------------------------------------------------------------------
# Panel (a): claimed vs measured dot strip
# ---------------------------------------------------------------------------

def _panel_claimed_vs_measured(ax, ranges: list[dict]) -> str:
    """Draw the claimed-vs-measured dot strip. Returns the PANDA range string
    (for the verify() CSV cross-check)."""
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(ranges) - 0.4)
    ax.invert_yaxis()  # TCGA at top, PanNuke at bottom
    ax.set_yticks(range(len(ranges)))
    ax.set_yticklabels([r["name"] for r in ranges], fontsize=_MIN_FONT)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=_MIN_FONT)
    ax.set_xlabel("Top-1 accuracy (%)", fontsize=_MIN_FONT)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=style.GRID_ALPHA)

    panda_range = None
    for i, r in enumerate(ranges):
        y = i
        # Measured range: solid line + endpoint markers in the dataset color.
        ax.plot([r["meas_lo"], r["meas_hi"]], [y, y], color=r["color"],
                linewidth=2.2, solid_capstyle="round", zorder=3)
        ax.plot([r["meas_lo"], r["meas_hi"]], [y, y], "o", color=r["color"],
                markersize=4.5, zorder=4)
        # Claimed range: black dashed band + open diamond at the midpoint.
        if r["claimed_lo"] is not None:
            c_lo, c_hi = r["claimed_lo"], r["claimed_hi"]
            ax.plot([c_lo, c_hi], [y, y], color="black", linestyle="--",
                    linewidth=1.4, zorder=2)
            ax.plot([(c_lo + c_hi) / 2], [y], "D", mfc="none", mec="black",
                    markersize=5, zorder=5)
        if r["name"] == "PANDA":
            panda_range = f"{r['meas_lo']:.2f}–{r['meas_hi']:.2f}"
    return panda_range


# ---------------------------------------------------------------------------
# Panel (b): the three "why" sub-axes (nested gridspec, equal height)
# ---------------------------------------------------------------------------

def _chip_empty_mask(ax, floor: float) -> None:
    """Sub-axis (i): empty-mask Dice crediting.

    Chip glyph (left), single-line label, single-line value.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Chip glyph: a small rounded box with a dot (X-ray style).
    box = FancyBboxPatch((0.02, 0.35), 0.10, 0.30,
                         boxstyle="round,pad=0.01,rounding_size=0.02",
                         facecolor=_LIGHT_GRAY, edgecolor=_NEUTRAL_GRAY,
                         linewidth=0.8, mutation_aspect=4)
    ax.add_patch(box)
    ax.plot(0.07, 0.50, "o", color=_OKABE_VERM, markersize=4, zorder=3)
    # Single-line label.
    ax.text(0.18, 0.62, "empty-mask Dice crediting",
            va="center", ha="left", fontsize=_MIN_FONT, color="black")
    # Single-line value.
    ax.text(0.18, 0.32, f"SIIM: {floor:.2f}% Dice floor",
            va="center", ha="left", fontsize=_MIN_FONT, color="black")


def _chip_leakage(ax) -> None:
    """Sub-axis (ii): patient-level leakage (GroupKFold glyphs).

    Chip glyph (left), single-line label, single-line value.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Glyph: two small clusters (leak vs split).
    for dx in [-0.02, 0.0, 0.02]:
        ax.plot(0.05 + dx, 0.58, "o", color=_OKABE_SKY, markersize=3, zorder=3)
    for dx in [-0.02, 0.0, 0.02]:
        ax.plot(0.09 + dx, 0.42, "o", color=_OKABE_GREEN, markersize=3, zorder=3)
    ax.plot([0.07, 0.07], [0.40, 0.60], color="black", linewidth=0.6,
            linestyle=":", zorder=2)
    # Single-line label.
    ax.text(0.18, 0.62, "patient-level leakage",
            va="center", ha="left", fontsize=_MIN_FONT, color="black")
    # Single-line value.
    ax.text(0.18, 0.32, "GroupKFold split",
            va="center", ha="left", fontsize=_MIN_FONT, color="black")


def _chip_task_interference(ax, static_acc: float, grad_acc: float) -> None:
    """Sub-axis (iii): task interference (45.39% → 29.04% arrow).

    Chip glyph (left), single-line label, single-line value.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Glyph: a small arrow.
    ax.annotate("", xy=(0.12, 0.50), xytext=(0.02, 0.50),
                arrowprops=dict(arrowstyle="-|>", color=_OKABE_PURPLE,
                                lw=1.5, mutation_scale=8))
    # Single-line label.
    ax.text(0.18, 0.62, "task interference (GradNorm)",
            va="center", ha="left", fontsize=_MIN_FONT, color="black")
    # Single-line value.
    ax.text(0.18, 0.32, f"PANDA: {static_acc:.2f}% → {grad_acc:.2f}% Acc",
            va="center", ha="left", fontsize=_MIN_FONT, color="black")


# ---------------------------------------------------------------------------
# Panel (c): audit protocol checklist
# ---------------------------------------------------------------------------

def _checklist_glyph(ax, x: float, y: float) -> None:
    """A small green checkmark in a rounded box."""
    box = FancyBboxPatch((x, y - 0.03), 0.06, 0.06,
                         boxstyle="round,pad=0.005,rounding_size=0.01",
                         facecolor=_OKABE_GREEN, edgecolor="none",
                         mutation_aspect=0.5)
    ax.add_patch(box)
    ax.plot([x + 0.012, x + 0.024, x + 0.048],
            [y + 0.012, y - 0.004, y + 0.030],
            color="white", linewidth=1.6, solid_capstyle="round", zorder=3)


def _panel_audit(ax) -> None:
    """Draw the three audit-protocol checklist items + the release line."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    items = [
        "group-aware splitting",
        "Dice evaluation convention",
        "95% Wilson intervals",
    ]
    ys = [0.86, 0.70, 0.54]
    for y, item in zip(ys, items):
        _checklist_glyph(ax, 0.06, y)
        ax.text(0.16, y + 0.015, item, va="center", ha="left",
                fontsize=_MIN_FONT, color="black")

    # Release line (two lines) + repo URL (two lines, small, bottom right).
    # Kept vertically separated from the checklist above and from each other.
    ax.text(0.06, 0.30, "benchmark + audit\nprotocol released",
            va="center", ha="left", fontsize=_MIN_FONT, color="black")
    ax.text(0.98, 0.02, "github.com/ApatheticMioz/\ncancer-pathology-dl",
            va="bottom", ha="right", fontsize=_MIN_FONT, color=_NEUTRAL_GRAY)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def _build_fig() -> tuple:
    """Build the banner and return ``(fig, axes_list, panda_range)``.

    ``axes_list`` is ``[ax_a, ax_b1, ax_b2, ax_b3, ax_c]`` where
    ``ax_b1..3`` are the three nested "Why" sub-axes.
    """
    matrix = results_matrix()
    ranges = _dataset_acc_ranges(matrix)
    floor = _siim_empty_floor(matrix)
    static_acc, grad_acc = _task_interference(matrix)

    w_in = _BANNER_CM[0] / _CM_PER_IN
    h_in = _BANNER_CM[1] / _CM_PER_IN
    fig = plt.figure(figsize=(w_in, h_in))
    gs = fig.add_gridspec(
        1, 3,
        left=0.14, right=0.98, top=0.80, bottom=0.20,
        wspace=0.15,
        width_ratios=[2, 4, 4],
    )
    ax_a = fig.add_subplot(gs[0, 0])

    # Middle panel: nested 3-row gridspec (equal height).
    gs_mid = gs[0, 1].subgridspec(3, 1, hspace=0.3)
    ax_b1 = fig.add_subplot(gs_mid[0, 0])
    ax_b2 = fig.add_subplot(gs_mid[1, 0])
    ax_b3 = fig.add_subplot(gs_mid[2, 0])

    ax_c = fig.add_subplot(gs[0, 2])

    # Panel titles (top of each panel).
    ax_a.set_title("Claimed vs measured", fontsize=9, fontweight="bold",
                   pad=6)
    ax_b1.set_title("Why", fontsize=9, fontweight="bold", pad=6)
    ax_c.set_title("Audit protocol", fontsize=9, fontweight="bold", pad=6)

    panda_range = _panel_claimed_vs_measured(ax_a, ranges)
    _chip_empty_mask(ax_b1, floor)
    _chip_leakage(ax_b2)
    _chip_task_interference(ax_b3, static_acc, grad_acc)
    _panel_audit(ax_c)

    return fig, [ax_a, ax_b1, ax_b2, ax_b3, ax_c], panda_range


def _save_banner(fig) -> list:
    """Save the banner as PDF (vector) + PNG (300 dpi) at the *full* figure
    size (``bbox_inches=None``) so the page aspect is the locked ~2.5:1.

    This deliberately deviates from :func:`paper.figures.style.save_figure`
    (which forces ``bbox_inches="tight"``) — see the module docstring.
    """
    out_dir = REPO_ROOT / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "graphical_abstract.pdf"
    png_path = out_dir / "graphical_abstract.png"
    # Temporarily lift the locked tight-bbox so the page = the full figure.
    prev_bbox = matplotlib.rcParams["savefig.bbox"]
    matplotlib.rcParams["savefig.bbox"] = None
    try:
        fig.savefig(pdf_path, bbox_inches=None)
        fig.savefig(png_path, bbox_inches=None, dpi=style.SAVE_DPI)
    finally:
        matplotlib.rcParams["savefig.bbox"] = prev_bbox
    return [pdf_path, png_path]


def build() -> list:
    """Render the graphical abstract and save PDF + PNG. Returns the paths."""
    fig, _, _ = _build_fig()
    paths = _save_banner(fig)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _pdf_page_size_pt(pdf_path: Path) -> tuple[float, float]:
    """Parse the PDF ``MediaBox`` and return ``(width_pt, height_pt)``."""
    data = pdf_path.read_bytes()
    m = re.search(
        rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]",
        data,
    )
    if not m:
        raise AssertionError(f"no MediaBox found in {pdf_path}")
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    return x1 - x0, y1 - y0


def _bbox_overlap(a, b, tol=0.0) -> bool:
    """True if two ``(x0, y0, x1, y1)`` bboxes overlap (with a small tol)."""
    return not (a[2] - tol <= b[0] or b[2] - tol <= a[0]
                or a[3] - tol <= b[1] or b[3] - tol <= a[1])


def _all_text_artists(ax) -> list:
    """Collect every *visible* Text artist in an axes (texts, title, tick
    labels, axis labels). Hidden artists (e.g. tick labels on an
    ``axis("off")`` sub-panel) are excluded."""
    texts = [t for t in ax.texts if t.get_visible()]
    if ax.title.get_visible() and ax.title.get_text():
        texts.append(ax.title)
    texts.extend(t for t in ax.xaxis.get_ticklabels() if t.get_visible())
    texts.extend(t for t in ax.yaxis.get_ticklabels() if t.get_visible())
    if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text():
        texts.append(ax.xaxis.label)
    if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text():
        texts.append(ax.yaxis.label)
    return texts


def verify() -> dict:
    """Verify the banner contract (no visual read-back).

    Asserts:
      * the PDF page aspect is within 5 % of 2.5:1 (2.375–2.625) and the cm
        size is within 13×5.2 to 13.4×5.4 cm;
      * the three panel titles do not overlap;
      * the PANDA measured range printed in panel (a) matches the CSV;
      * every Text artist's window extent is inside the figure bbox;
      * no two Text artists in the same axes overlap (1 px tolerance).
    Returns a summary dict.
    """
    fig, axes, panda_range = _build_fig()
    fig.canvas.draw()

    # --- (1) PDF page size / aspect -------------------------------------
    tmp = Path(tempfile.mktemp(suffix=".pdf"))
    prev_bbox = matplotlib.rcParams["savefig.bbox"]
    matplotlib.rcParams["savefig.bbox"] = None
    try:
        fig.savefig(tmp, bbox_inches=None)
    finally:
        matplotlib.rcParams["savefig.bbox"] = prev_bbox
    w_pt, h_pt = _pdf_page_size_pt(tmp)
    tmp.unlink(missing_ok=True)
    w_cm, h_cm = w_pt / 72.0 * _CM_PER_IN, h_pt / 72.0 * _CM_PER_IN
    aspect = w_cm / h_cm
    assert 2.375 <= aspect <= 2.625, (
        f"banner aspect {aspect:.3f} not within 5% of 2.5:1"
    )
    assert 13.0 <= w_cm <= 13.4, f"banner width {w_cm:.2f}cm outside 13.0–13.4"
    assert 5.2 <= h_cm <= 5.4, f"banner height {h_cm:.2f}cm outside 5.2–5.4"

    # --- (2) panel titles do not overlap --------------------------------
    renderer = fig.canvas.get_renderer()
    titles = [axes[0], axes[1], axes[4]]  # ax_a, ax_b1, ax_c
    tboxes = []
    for ax in titles:
        t = ax.title
        bb = t.get_window_extent(renderer)
        tboxes.append((float(bb.x0), float(bb.y0),
                       float(bb.x1), float(bb.y1)))
    overlaps = []
    for i in range(len(tboxes)):
        for j in range(i + 1, len(tboxes)):
            if _bbox_overlap(tboxes[i], tboxes[j]):
                overlaps.append((i, j))
    assert not overlaps, f"panel titles overlap: {overlaps}"

    # --- (3) PANDA range matches the CSV --------------------------------
    matrix = results_matrix()
    panda = matrix[matrix["Dataset"] == "PANDA"]["Accuracy (%)"]
    expected = f"{float(panda.min()):.2f}–{float(panda.max()):.2f}"
    assert panda_range == expected, (
        f"printed PANDA range {panda_range!r} != CSV {expected!r}"
    )

    # --- (4) all Text extents inside the figure bbox ---------------------
    # get_window_extent returns display pixels; the figure's pixel extent
    # is (figwidth × renderer.dpi) × (figheight × renderer.dpi) with the
    # origin at the bottom-left corner.
    fig_w_px = fig.get_figwidth() * renderer.dpi
    fig_h_px = fig.get_figheight() * renderer.dpi
    outside_texts = []
    for ax in fig.get_axes():
        for t in _all_text_artists(ax):
            bb = t.get_window_extent(renderer)
            if not (0 <= bb.x0 and bb.y0 >= 0
                    and bb.x1 <= fig_w_px and bb.y1 <= fig_h_px):
                outside_texts.append(t.get_text())
    assert not outside_texts, (
        f"texts outside figure bbox: {outside_texts}"
    )

    # --- (5) no two Text artists in the same axes overlap (1 px tol) ----
    text_overlaps = []
    for ax in fig.get_axes():
        texts = _all_text_artists(ax)
        bboxes = [(t, t.get_window_extent(renderer)) for t in texts]
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                t1, bb1 = bboxes[i]
                t2, bb2 = bboxes[j]
                if _bbox_overlap(
                    (bb1.x0, bb1.y0, bb1.x1, bb1.y1),
                    (bb2.x0, bb2.y0, bb2.x1, bb2.y1),
                    tol=1.0,
                ):
                    text_overlaps.append(
                        f"{t1.get_text()!r} ∩ {t2.get_text()!r}"
                    )
    assert not text_overlaps, (
        f"text overlaps within axes: {text_overlaps}"
    )

    plt.close(fig)
    return {
        "ga_pdf_size_pt": (round(w_pt, 3), round(h_pt, 3)),
        "ga_pdf_size_cm": (round(w_cm, 3), round(h_cm, 3)),
        "ga_aspect": round(aspect, 4),
        "ga_n_title_overlaps": len(overlaps),
        "ga_panda_range": panda_range,
        "ga_panda_range_matches_csv": True,
        "ga_n_text_overlaps": len(text_overlaps),
        "ga_n_texts_outside_bbox": len(outside_texts),
    }


if __name__ == "__main__":
    for p in build():
        print(f"Saved: {p}")
