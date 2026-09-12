"""Figure 2 — The Empty-Mask Background-Dice Inflation Fallacy.

Port of the original ``paper/generate_figures.py::generate_figure_2`` onto the
``paper.figures`` package. Three panels, unchanged in layout and content:

* (a) a lesion-free SIIM slice (``|Y| = 0``) — the empty-credit convention
  awards it Dice = 1.0;
* (b) a lesion-bearing SIIM slice with the ground-truth mask overlaid — a
  collapsed (all-empty) model scores Dice = 0% on it;
* (c) the slice-averaged Dice inflation curve.

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


def _build_fig():
    """Build the Figure 2 ``Figure`` and return ``(fig, axes)``.

    Split out from :func:`build` so the layout can be inspected programmatically
    (see :func:`verify`) without writing files.
    """
    curve = dice_degeneracy()

    img_neg = np.array(Image.open(_NEG_IMG).convert("L"))
    img_pos = np.array(Image.open(_POS_IMG).convert("L"))
    mask_pos = np.array(Image.open(_POS_MASK).convert("L"))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), dpi=300)

    # --- Panel (a): true-negative (lesion-free) slice -------------------
    axes[0].imshow(img_neg, cmap="gray")
    # Single-line title. The secondary "Dice = 1.0 (empty-credit)" info lives
    # in the in-image annotation box below, so the title never collides with
    # the image or a neighbouring panel's title.
    axes[0].set_title(r"(a) Lesion-Free Slice ($|Y|=0$)", fontweight="bold")
    textstr = (
        "Ground Truth: Empty ($|Y|=0$)\n"
        "Prediction: Empty ($|\\hat{Y}|=0$)\n"
        r"$\mathbf{Dice(Y, \hat{Y}) = 1.0}$"
        "\n(awarded on the 77.7% lesion-free\n"
        "fraction of the validation split)"
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.85,
                 edgecolor="black", linewidth=0.8)
    axes[0].text(0.05, 0.08, textstr, transform=axes[0].transAxes,
                 fontsize=7.5, verticalalignment="bottom", bbox=props)
    axes[0].axis("off")

    # --- Panel (b): lesion-bearing slice + GT mask overlay -------------
    axes[1].imshow(img_pos, cmap="gray")
    mask_overlay = np.zeros((*img_pos.shape, 4), dtype=float)
    mask_overlay[mask_pos > 0] = [0.0, 1.0, 0.2, 0.45]  # semi-transparent green
    axes[1].imshow(mask_overlay)
    axes[1].contour(mask_pos > 0, colors=["lime"], linewidths=1.2)
    # Single-line title; the "Dice = 0%" detail stays in the in-image box.
    axes[1].set_title(r"(b) Lesion-Bearing Slice ($|Y|>0$)", fontweight="bold")
    textstr_b = (
        "Pneumothorax Pleural Lesion\n"
        "Ground-Truth Boundary (Green)\n"
        "All-Empty Prediction\n"
        r"$\mathbf{Dice = 0\%}$"
        "\n(invisible to the slice-averaged score)"
    )
    props_b = dict(boxstyle="round", facecolor="white", alpha=0.85,
                   edgecolor="black", linewidth=0.8)
    axes[1].text(0.05, 0.08, textstr_b, transform=axes[1].transAxes,
                 fontsize=7.5, verticalalignment="bottom", bbox=props_b)
    axes[1].axis("off")

    # --- Panel (c): slice-averaged Dice inflation curve ----------------
    # Curve points come from the CSV (empty_slice_ratio -> reported macro Dice).
    rho_pct = curve["empty_slice_ratio"].to_numpy() * 100.0
    reported_pct = curve["reported_macro_dice_pct"].to_numpy()

    # The inflation curve (SIIM color).
    axes[2].plot(rho_pct, reported_pct, color=style.DATASET_COLORS["SIIM"],
                 linewidth=1.2,
                 label=r"$\overline{\mathrm{Dice}}_{\mathrm{all}} = \rho + (1-\rho)\, d_{fg}$")

    # Published claim (reference / claimed style: black dashed).
    axes[2].axhline(y=_PUBLISHED_PCT, color="black", linestyle="--",
                    linewidth=1.2, label=f"Published Claim ({_PUBLISHED_PCT:.1f}% Dice)")

    # Observed empty-slice fraction.
    axes[2].axvline(x=_RHO_HAT * 100.0, color=style.NULL_GRAY, linestyle=":",
                    linewidth=1.2, label=r"Empty-Slice Fraction ($\hat{\rho}=77.7\%$)")

    # Analytic illustration: a model with genuine foreground Dice 77.74%
    # would report rho_hat*1.0 + (1-rho_hat)*0.7774 = 95.04%. Small marker +
    # short label only (no arrow) so it does not cross the lines; the full
    # explanation belongs in the figure caption.
    siim_inflated = _RHO_HAT * 1.0 + (1.0 - _RHO_HAT) * (_FG_DICE_PCT / 100.0)
    axes[2].plot(_RHO_HAT * 100.0, siim_inflated * 100.0,
                 marker=style.CLAIMED_STYLE["marker"],
                 mfc=style.CLAIMED_STYLE["mfc"], mec=style.CLAIMED_STYLE["mec"],
                 color=style.CLAIMED_STYLE["color"], markersize=6, linestyle="none")
    axes[2].annotate(f"{siim_inflated * 100:.1f}%",
                     xy=(_RHO_HAT * 100.0, siim_inflated * 100),
                     xytext=(7, 3), textcoords="offset points",
                     fontsize=7, color="black")

    # Measured floor: all-empty predictions (77.74%). Marker + short label;
    # included in the legend.
    axes[2].plot(_RHO_HAT * 100.0, _FLOOR_PCT, marker="s", color="darkred",
                 markersize=6, linestyle="none",
                 label=f"Measured Floor (all-empty, {_FLOOR_PCT:.1f}%)")
    axes[2].annotate(f"{_FLOOR_PCT:.1f}%",
                     xy=(_RHO_HAT * 100.0, _FLOOR_PCT),
                     xytext=(7, -3), textcoords="offset points",
                     fontsize=7, color="darkred")

    # Y-limits span the full 0-100 range so the curve's anchor at
    # (rho=0, 44.08) is visible (the previous [70,101] window hid it).
    axes[2].set_xlim([0, 100])
    axes[2].set_ylim([0, 100])
    # Add an explicit tick at the curve's 44.08 anchor and label it.
    axes[2].set_yticks([0, 20, 40, 44.08, 60, 80, 100])
    axes[2].set_yticklabels(["0", "20", "40", "44.08", "60", "80", "100"])
    axes[2].annotate("curve anchor", xy=(0, 44.08), xytext=(6, -11),
                     textcoords="offset points", fontsize=6.5, color="black")

    axes[2].set_xlabel(r"Empty-Slice Fraction $\rho$ (%)", fontweight="bold")
    axes[2].set_ylabel("Reported Macroscopic Dice (%)", fontweight="bold")
    axes[2].set_title("(c) Slice-Averaged Dice vs. Empty-Slice Fraction",
                      fontweight="bold")

    # Single frameless legend with all four entries, anchored in the
    # lower-left corner — the only region clear of every line: the
    # published-claim line at y=99 occupies the upper band, and the curve
    # rises from (0, 44.08), so the lower-left (y < ~40) is empty.
    axes[2].legend(loc="lower left", fontsize=7)
    axes[2].grid(True, linestyle="--", alpha=style.GRID_ALPHA)

    plt.tight_layout()
    return fig, axes


def build() -> list:
    """Render Figure 2 and save PDF + PNG. Returns the written paths."""
    fig, _ = _build_fig()
    out_dir = REPO_ROOT / "paper"
    return style.save_figure(fig, out_dir, "fig2_empty_mask_dice")


def _bbox_overlap(a, b, tol=0.0) -> bool:
    """True if two ``(x0, y0, x1, y1)`` bboxes overlap (with a small tol)."""
    return not (a[2] - tol <= b[0] or b[2] - tol <= a[0]
                or a[3] - tol <= b[1] or b[3] - tol <= a[1])


def verify() -> dict:
    """Programmatically verify the Figure 2 layout (no visual read-back).

    Draws the canvas and checks, for panel (c):
      * the legend's bounding box does not overlap any data line (the
        published-claim line, the inflation curve, the empty-slice
        fraction line, or the floor/analytic markers);
      * the legend does not overlap the y-axis tick labels;
      * the y-limits expose the curve's 44.08 anchor.
    Also reports the final axes limits and the legend bbox (in data coords).
    Returns a dict; raises ``AssertionError`` on any overlap.
    """
    fig, axes = _build_fig()
    ax = axes[2]
    fig.canvas.draw()  # force layout so bboxes are valid

    renderer = fig.canvas.get_renderer()
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
    # The inflation curve is the first labelled line; the axhline/axvline are
    # also lines. Collect their data extents.
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
    return result


if __name__ == "__main__":
    for p in build():
        print(f"Saved: {p}")
