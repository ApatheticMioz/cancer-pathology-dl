"""Figure 2 — The Empty-Mask Background-Dice Inflation Fallacy.

Port of the original ``paper/generate_figures.py::generate_figure_2`` onto the
``paper.figures`` package. Three panels:

* (a) a lesion-free SIIM slice (``|Y| = 0``) — the empty-credit convention
  awards it Dice = 1.0;
* (b) a lesion-bearing SIIM slice with the ground-truth mask overlaid — a
  collapsed (all-empty) model scores Dice = 0% on it;
* (c) the slice-averaged Dice inflation curve.

Layout (enlarged X-ray panels)
------------------------------
The two SIIM radiograph tiles (a)/(b) are the *hero* of the figure: they must
be large enough to read the pathology. The canvas is therefore a **two-row**
layout:

* **Top row** — the two square radiograph tiles (a) and (b), side by side,
  given the larger share of the figure height so each tile renders ~6 cm tall
  in the compiled paper (previously ~3.8 cm, too small to read).
* **Bottom row** — the degeneracy curve (c), full-width but shorter, with its
  own (larger) left margin so the rotated y-axis label fits inside the figure.

The figure's *physical* width is set to ``0.98 * textwidth`` (the elsarticle
preprint ``textwidth`` is 384 pt = 5.333 in) so that, when the paper prints it
with ``\\includegraphics[width=0.98\\textwidth]``, the print scale is ~1.0 and
every :mod:`paper.figures.style` font (>= 7 pt) prints at >= 7 pt. The in-image
annotation boxes are slimmed (shorter strings, one font step smaller) so they
spend fewer pixels over the radiograph.

What changed versus the original:

* All styling comes from :mod:`paper.figures.style` (``apply()`` once at
  import; the null/claimed reference styling uses the locked
  ``CLAIMED_STYLE`` and ``NULL_GRAY``).
* The inflation curve in panel (c) is **read from**
  :func:`paper.figures.loaders.dice_degeneracy` (the
  ``paper/dice_degeneracy_curve.csv`` points), not hard-coded.
* The pinned constants (empty-slice fraction ``rho_hat``, the all-empty
  floor, and the published claim) carry provenance comments pointing at
  ``scripts/verify_siim_floor.py``.

Output: ``paper/fig2_empty_mask_dice.pdf`` + ``paper/fig2_empty_mask_dice.png``
(via :func:`paper.figures.style.save_figure`).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from . import style
from .loaders import REPO_ROOT, dice_degeneracy

# Apply the locked style contract once for this module.
style.apply()

# ---------------------------------------------------------------------------
# Pinned image paths (local data artifacts — NOT CSV-sourced).
# ---------------------------------------------------------------------------

#: Lesion-free (empty ground-truth) SIIM slice. Pinned local path.
_NEG_IMG = REPO_ROOT / "data" / "SIIM" / "preprocessed" / "images" / "1.2.276.0.7230010.3.1.4.8323329.1000.1517875165.878027.png"
#: Lesion-bearing SIIM slice. Pinned local path.
_POS_IMG = REPO_ROOT / "data" / "SIIM" / "preprocessed" / "images" / "1.2.276.0.7230010.3.1.4.8323329.10005.1517875220.958951.png"
#: Ground-truth mask for the lesion-bearing slice. Pinned local path.
_POS_MASK = REPO_ROOT / "data" / "SIIM" / "preprocessed" / "masks" / "1.2.276.0.7230010.3.1.4.8323329.10005.1517875220.958951_mask.png"

# ---------------------------------------------------------------------------
# Pinned constants (with provenance).
# ---------------------------------------------------------------------------

# Empty-slice fraction of the SIIM validation split: 1659 empty masks out of
# 2135 validation slices. Provenance: scripts/verify_siim_floor.py replicates
# the exact parse_siim + StratifiedShuffleSplit (seed 42, test_size 0.2)
# pipeline and pins n = 2135 and the empty-mask count; the all-empty
# batch-mean Dice it reproduces is 0.7774375410222295.
_RHO_HAT = 1659.0 / 2135.0  # = 0.77700...

# The all-empty-prediction floor: every one of the four SIIM runs (05, 06,
# 11, 12) emits empty masks on every validation slice, so the reported
# slice-averaged Dice is exactly the empty-slice fraction = 77.74%.
# Provenance: scripts/verify_siim_floor.py (EXPECTED_FLOOR =
# 0.7774375410222295) and the SIIM rows of paper/paper_results_matrix_with_ci.csv
# (Macro Dice = 77.74 for all four runs).
_FLOOR_PCT = 77.74

# The published claim being refuted: 99.0% macro Dice. Provenance: the
# published multi-task pneumothorax result (rhanoui2025multitask) cited in
# the manuscript; also the "Published Claim (99.0% Dice)" reference line in
# the original figure.
_PUBLISHED_PCT = 99.0

# Foreground Dice used for the analytic illustration point: a model with a
# genuine foreground Dice of 77.74% would report
#   rho_hat * 1.0 + (1 - rho_hat) * 0.7774 = 95.04%
# at the observed empty-slice fraction. Provenance: manuscript Eq. (slice
# Dice) and scripts/verify_siim_floor.py.
_FG_DICE_PCT = 77.74

# ---------------------------------------------------------------------------
# Locked figure geometry (layout parameters — not data values).
# ---------------------------------------------------------------------------

# The elsarticle preprint textwidth is 384 pt (see elsarticle.cls). The paper
# prints this figure with \includegraphics[width=0.98\textwidth], i.e. at
# 0.98 * 384 pt = 376.3 pt = 5.227 in. Setting the figure's *physical* width
# to that same value makes the print scale ~1.0, so every style.py font
# (>= 7 pt) prints at >= 7 pt.
_TEXTWIDTH_IN = 384.0 / 72.0
_PRINT_W_IN = 0.98 * _TEXTWIDTH_IN

#: Figure height (inches). The top (image) row is given the larger share of
#: this height so the radiograph tiles render ~6 cm tall in the paper.
_FIG_H_IN = 5.6

# Top row: the two radiograph tiles. Small left/right margins so the columns
# are wide (the square tile is width-limited). The row spans the upper ~44%
# of the figure height.
_IMG_LEFT, _IMG_RIGHT = 0.05, 0.95
_IMG_TOP, _IMG_BOT = 0.90, 0.46
_IMG_WSPACE = 0.02

# Bottom row: the degeneracy curve. A larger left margin than the image row
# so the rotated y-axis label fits inside the figure (and the tight bbox
# stays <= the print width, keeping the print scale >= 1.0). The row is
# tall enough that the 4-entry frameless legend (lower-left) stays below the
# curve's 44.08 anchor, so it never overlaps the data lines.
_C_LEFT, _C_RIGHT = 0.12, 0.95
_C_TOP, _C_BOT = 0.42, 0.08

# In-image annotation font: one step smaller than the previous 7.5 pt so the
# boxes spend fewer pixels over the radiograph. At the ~1.0 print scale this
# still prints at >= 7 pt (the style.py minimum).
_ANNOT_FONT = 7.0


def _build_fig():
    """Build the Figure 2 ``Figure`` and return ``(fig, axes)``.

    Two-row layout: the two radiograph tiles (a)/(b) on top (the larger
    height share), the degeneracy curve (c) below. Split out from
    :func:`build` so the layout can be inspected programmatically (see
    :func:`verify`) without writing files.
    """
    curve = dice_degeneracy()

    img_neg = np.array(Image.open(_NEG_IMG).convert("L"))
    img_pos = np.array(Image.open(_POS_IMG).convert("L"))
    mask_pos = np.array(Image.open(_POS_MASK).convert("L"))

    fig = plt.figure(figsize=(_PRINT_W_IN, _FIG_H_IN))

    # --- Top row: the two radiograph tiles (a) and (b) -------------------
    gs_img = fig.add_gridspec(
        1, 2,
        left=_IMG_LEFT, right=_IMG_RIGHT,
        top=_IMG_TOP, bottom=_IMG_BOT,
        wspace=_IMG_WSPACE,
    )
    ax_a = fig.add_subplot(gs_img[0, 0])
    ax_b = fig.add_subplot(gs_img[0, 1])

    # --- Bottom row: the degeneracy curve (c) ----------------------------
    gs_c = fig.add_gridspec(
        1, 1,
        left=_C_LEFT, right=_C_RIGHT,
        top=_C_TOP, bottom=_C_BOT,
    )
    ax_c = fig.add_subplot(gs_c[0, 0])

    # --- Panel (a): true-negative (lesion-free) slice -------------------
    ax_a.imshow(img_neg, cmap="gray")
    # Clear the image-axis tick labels so they do not leak into the tight
    # bbox (axis("off") hides the spines but the tick labels can still
    # contribute to the saved extent).
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.axis("off")
    # Single-line title. The "Dice = 1.0 (empty-credit)" detail lives in the
    # slim in-image annotation box below, so the title never collides with
    # the image or a neighbouring panel's title.
    ax_a.set_title(r"(a) Lesion-Free Slice ($|Y|=0$)", fontweight="bold")
    # Slim annotation box: three short lines (down from five), one font step
    # smaller, so it spends fewer pixels over the radiograph.
    textstr = (
        r"GT: empty ($|Y|=0$)"
        "\n"
        r"Pred: empty ($|\hat{Y}|=0$)"
        "\n"
        r"$\mathbf{Dice = 1.0}$"
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.85,
                 edgecolor="black", linewidth=0.8)
    ax_a.text(0.05, 0.05, textstr, transform=ax_a.transAxes,
              fontsize=_ANNOT_FONT, verticalalignment="bottom", bbox=props)

    # --- Panel (b): lesion-bearing slice + GT mask overlay -------------
    ax_b.imshow(img_pos, cmap="gray")
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    mask_overlay = np.zeros((*img_pos.shape, 4), dtype=float)
    mask_overlay[mask_pos > 0] = [0.0, 1.0, 0.2, 0.45]  # semi-transparent green
    ax_b.imshow(mask_overlay)
    ax_b.contour(mask_pos > 0, colors=["lime"], linewidths=1.2)
    ax_b.axis("off")
    # Single-line title; the "Dice = 0%" detail stays in the slim in-image box.
    ax_b.set_title(r"(b) Lesion-Bearing Slice ($|Y|>0$)", fontweight="bold")
    textstr_b = (
        "GT lesion (green)"
        "\n"
        "Pred: all-empty"
        "\n"
        r"$\mathbf{Dice = 0\%}$"
    )
    props_b = dict(boxstyle="round", facecolor="white", alpha=0.85,
                   edgecolor="black", linewidth=0.8)
    ax_b.text(0.05, 0.05, textstr_b, transform=ax_b.transAxes,
              fontsize=_ANNOT_FONT, verticalalignment="bottom", bbox=props_b)

    # --- Panel (c): slice-averaged Dice inflation curve ----------------
    # Curve points come from the CSV (empty_slice_ratio -> reported macro Dice).
    rho_pct = curve["empty_slice_ratio"].to_numpy() * 100.0
    reported_pct = curve["reported_macro_dice_pct"].to_numpy()

    # The inflation curve (SIIM color).
    ax_c.plot(rho_pct, reported_pct, color=style.DATASET_COLORS["SIIM"],
              linewidth=1.2,
              label=r"$\overline{\mathrm{Dice}}_{\mathrm{all}} = \rho + (1-\rho)\, d_{fg}$")

    # Published claim (reference / claimed style: black dashed).
    ax_c.axhline(y=_PUBLISHED_PCT, color="black", linestyle="--",
                 linewidth=1.2, label=f"Published Claim ({_PUBLISHED_PCT:.1f}% Dice)")

    # Observed empty-slice fraction.
    ax_c.axvline(x=_RHO_HAT * 100.0, color=style.NULL_GRAY, linestyle=":",
                 linewidth=1.2, label=r"Empty-Slice Fraction ($\hat{\rho}=77.7\%$)")

    # Analytic illustration: a model with genuine foreground Dice 77.74%
    # would report rho_hat*1.0 + (1-rho_hat)*0.7774 = 95.04%. Small marker +
    # short label only (no arrow) so it does not cross the lines; the full
    # explanation belongs in the figure caption.
    siim_inflated = _RHO_HAT * 1.0 + (1.0 - _RHO_HAT) * (_FG_DICE_PCT / 100.0)
    ax_c.plot(_RHO_HAT * 100.0, siim_inflated * 100.0,
              marker=style.CLAIMED_STYLE["marker"],
              mfc=style.CLAIMED_STYLE["mfc"], mec=style.CLAIMED_STYLE["mec"],
              color=style.CLAIMED_STYLE["color"], markersize=6, linestyle="none")
    ax_c.annotate(f"{siim_inflated * 100:.1f}%",
                  xy=(_RHO_HAT * 100.0, siim_inflated * 100),
                  xytext=(7, 3), textcoords="offset points",
                  fontsize=7, color="black")

    # Measured floor: all-empty predictions (77.74%). Marker + short label;
    # included in the legend.
    ax_c.plot(_RHO_HAT * 100.0, _FLOOR_PCT, marker="s", color="darkred",
              markersize=6, linestyle="none",
              label=f"Measured Floor (all-empty, {_FLOOR_PCT:.1f}%)")
    ax_c.annotate(f"{_FLOOR_PCT:.1f}%",
                  xy=(_RHO_HAT * 100.0, _FLOOR_PCT),
                  xytext=(7, -3), textcoords="offset points",
                  fontsize=7, color="darkred")

    # Y-limits span the full 0-100 range so the curve's anchor at
    # (rho=0, 44.08) is visible (the previous [70,101] window hid it).
    ax_c.set_xlim([0, 100])
    ax_c.set_ylim([0, 100])
    # Add an explicit tick at the curve's 44.08 anchor and label it. The
    # regular 40 tick is dropped (it sits only 4.08 units below the 44.08
    # anchor, so its 7 pt label would collide with the anchor's in this
    # shorter, wider panel); the 44.08 anchor itself stays visible.
    ax_c.set_yticks([0, 20, 44.08, 60, 80, 100])
    ax_c.set_yticklabels(["0", "20", "44.08", "60", "80", "100"])
    ax_c.annotate("curve anchor", xy=(0, 44.08), xytext=(6, -11),
                  textcoords="offset points", fontsize=6.5, color="black")

    ax_c.set_xlabel(r"Empty-Slice Fraction $\rho$ (%)", fontweight="bold")
    ax_c.set_ylabel("Reported Macroscopic Dice (%)", fontweight="bold")
    ax_c.set_title("(c) Slice-Averaged Dice vs. Empty-Slice Fraction",
                   fontweight="bold")

    # Single frameless legend with all four entries, anchored in the
    # lower-left corner — the only region clear of every line: the
    # published-claim line at y=99 occupies the upper band, and the curve
    # rises from (0, 44.08), so the lower-left (y < ~40) is empty.
    ax_c.legend(loc="lower left", fontsize=7)
    ax_c.grid(True, linestyle="--", alpha=style.GRID_ALPHA)

    return fig, [ax_a, ax_b, ax_c]


def build() -> list:
    """Render Figure 2 and save PDF + PNG. Returns the written paths."""
    fig, _ = _build_fig()
    out_dir = REPO_ROOT / "paper"
    return style.save_figure(fig, out_dir, "fig2_empty_mask_dice")


def _bbox_overlap(a, b, tol=0.0) -> bool:
    """True if two ``(x0, y0, x1, y1)`` bboxes overlap (with a small tol)."""
    return not (a[2] - tol <= b[0] or b[2] - tol <= a[0]
                or a[3] - tol <= b[1] or b[3] - tol <= a[1])


def _all_text_artists(ax) -> list:
    """Collect every *visible* Text artist in an axes (texts, title, tick
    labels, axis labels). Hidden artists (e.g. tick labels on an
    ``axis("off")`` sub-panel) are excluded.

    Ported verbatim from :mod:`paper.figures.graphical_abstract` so the
    rendered-extent assertions are identical across figures.
    """
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
    """Programmatically verify the Figure 2 layout (no visual read-back).

    Draws the canvas and checks:

    * **Panel (c) data contract** (original): the legend's bounding box does
      not overlap any data line (the published-claim line, the inflation
      curve, the empty-slice-fraction line, or the floor/analytic markers);
      the legend does not overlap the y-axis tick labels; and the y-limits
      expose the curve's 44.08 anchor.
    * **Rendered-extent contract** (same assertions as
      :mod:`paper.figures.graphical_abstract`): every visible Text artist's
      window extent is *fully inside the figure bbox*, and no two Text
      artists in the same axes overlap (1 px tolerance). This catches the
      enlarged tiles' in-image annotation boxes, the single-line panel
      titles, and panel (c)'s labels/ticks/legend.

    Also reports the final axes limits, the legend bbox (in data coords),
    and the figure's rendered size. Returns a dict; raises ``AssertionError``
    on any overlap or out-of-bounds text.
    """
    fig, axes = _build_fig()
    ax = axes[2]
    fig.canvas.draw()  # force layout so bbox/window extents are valid

    renderer = fig.canvas.get_renderer()

    # --- (1) Panel (c): legend vs data lines ----------------------------
    legend = ax.get_legend()
    leg_bbox = legend.get_window_extent(renderer)
    leg_data = ax.transData.inverted().transform(leg_bbox)
    leg_data = (float(leg_data[0][0]), float(leg_data[0][1]),
                float(leg_data[1][0]), float(leg_data[1][1]))

    # Data lines to check the legend against (in data coords).
    lines = []
    for ln in ax.get_lines():
        if ln.get_label() not in (".none", ""):
            lines.append(ln)
    line_bboxes = []
    for ln in lines:
        xs, ys = ln.get_xdata(), ln.get_ydata()
        if len(xs) == 0:
            continue
        line_bboxes.append((float(np.min(xs)), float(np.min(ys)),
                            float(np.max(xs)), float(np.max(ys))))

    overlaps = []
    for i, lb in enumerate(line_bboxes):
        if _bbox_overlap(leg_data, lb):
            overlaps.append((i, lb))

    ylim = ax.get_ylim()
    result = {
        "fig2c_ylim": (float(ylim[0]), float(ylim[1])),
        "fig2c_xlim": tuple(float(v) for v in ax.get_xlim()),
        "fig2c_legend_bbox_data": leg_data,
        "fig2c_legend_n_entries": len(legend.get_texts()),
        "fig2c_curve_anchor_visible": bool(ylim[0] <= 44.08 <= ylim[1]),
        "fig2c_legend_line_overlaps": overlaps,
    }
    if overlaps:
        raise AssertionError(
            f"fig2c legend overlaps {len(overlaps)} data line(s): {overlaps}"
        )
    if not result["fig2c_curve_anchor_visible"]:
        raise AssertionError(
            f"fig2c y-limits {ylim} do not expose the 44.08 curve anchor"
        )

    # --- (2) All Text artists fully inside the figure bbox --------------
    # get_window_extent returns display pixels; the figure's pixel extent is
    # (figwidth * renderer.dpi) x (figheight * renderer.dpi) with the origin
    # at the bottom-left corner.
    fig_w_px = fig.get_figwidth() * renderer.dpi
    fig_h_px = fig.get_figheight() * renderer.dpi
    outside_texts = []
    for ax_i in fig.get_axes():
        for t in _all_text_artists(ax_i):
            bb = t.get_window_extent(renderer)
            if not (0 <= bb.x0 and bb.y0 >= 0
                    and bb.x1 <= fig_w_px and bb.y1 <= fig_h_px):
                outside_texts.append(t.get_text())
    if outside_texts:
        raise AssertionError(f"texts outside figure bbox: {outside_texts}")

    # --- (3) No two Text artists in the same axes overlap (1 px tol) ----
    text_overlaps = []
    for ax_i in fig.get_axes():
        texts = _all_text_artists(ax_i)
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
    if text_overlaps:
        raise AssertionError(f"text overlaps within axes: {text_overlaps}")

    result["fig2_n_texts_outside_bbox"] = len(outside_texts)
    result["fig2_n_text_overlaps"] = len(text_overlaps)
    result["fig2_fig_size_in"] = (round(fig.get_figwidth(), 3),
                                  round(fig.get_figheight(), 3))
    return result


if __name__ == "__main__":
    for p in build():
        print(f"Saved: {p}")
