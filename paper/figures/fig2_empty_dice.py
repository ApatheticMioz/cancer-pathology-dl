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


def build() -> list:
    """Render Figure 2 and save PDF + PNG. Returns the written paths."""
    curve = dice_degeneracy()

    img_neg = np.array(Image.open(_NEG_IMG).convert("L"))
    img_pos = np.array(Image.open(_POS_IMG).convert("L"))
    mask_pos = np.array(Image.open(_POS_MASK).convert("L"))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), dpi=300)

    # --- Panel (a): true-negative (lesion-free) slice -------------------
    axes[0].imshow(img_neg, cmap="gray")
    axes[0].set_title(
        r"(a) Lesion-Free Slice ($|Y|=0$)"
        r"\nDice $\equiv 1.0$ (Empty-Credit Convention)",
        fontweight="bold",
    )
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
    axes[1].set_title(
        r"(b) Lesion-Bearing Slice ($|Y|>0$)"
        r"\nCollapsed Model Scores Dice $= 0\%$",
        fontweight="bold",
    )
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
    # would report rho_hat*1.0 + (1-rho_hat)*0.7774 = 95.04%.
    siim_inflated = _RHO_HAT * 1.0 + (1.0 - _RHO_HAT) * (_FG_DICE_PCT / 100.0)
    axes[2].plot(_RHO_HAT * 100.0, siim_inflated * 100.0,
                 marker=style.CLAIMED_STYLE["marker"],
                 mfc=style.CLAIMED_STYLE["mfc"], mec=style.CLAIMED_STYLE["mec"],
                 color=style.CLAIMED_STYLE["color"], markersize=7, linestyle="none")
    axes[2].annotate(
        "Analytic illustration:\n"
        r"$d_{fg}=77.74\%$ reports 95.04%",
        xy=(_RHO_HAT * 100.0, siim_inflated * 100),
        xytext=(50, 80),
        arrowprops=dict(facecolor="darkblue", shrink=0.08, width=1, headwidth=5),
        fontweight="bold", fontsize=8, color="darkblue",
    )

    # Measured floor: all-empty predictions (77.74%).
    axes[2].plot(_RHO_HAT * 100.0, _FLOOR_PCT, marker="s", color="darkred",
                 markersize=7, linestyle="none")
    axes[2].annotate(
        "Measured floor:\nall-empty predictions\n(77.74%)",
        xy=(_RHO_HAT * 100.0, _FLOOR_PCT),
        xytext=(45, 71.5),
        arrowprops=dict(facecolor="darkred", shrink=0.08, width=1, headwidth=5),
        fontweight="bold", fontsize=8, color="darkred",
    )

    axes[2].set_xlabel(r"Empty-Slice Fraction $\rho$ (%)", fontweight="bold")
    axes[2].set_ylabel("Reported Macroscopic Dice (%)", fontweight="bold")
    axes[2].set_title("(c) Slice-Averaged Dice vs.\nEmpty-Slice Fraction", fontweight="bold")
    axes[2].set_xlim([0, 100])
    axes[2].set_ylim([70, 101])
    axes[2].legend(loc="upper left", fontsize=7)
    axes[2].grid(True, linestyle="--", alpha=style.GRID_ALPHA)

    plt.tight_layout()
    out_dir = REPO_ROOT / "paper"
    return style.save_figure(fig, out_dir, "fig2_empty_mask_dice")


if __name__ == "__main__":
    for p in build():
        print(f"Saved: {p}")
