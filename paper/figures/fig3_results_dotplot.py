"""Figure 3 — 26-run claimed-vs-measured dot plot with 95% Wilson CIs.

The paper's centerpiece figure. Two side-by-side panels share a common
categorical y-axis of 26 rows (one row per run, in CSV order, which is
already grouped into the five experiment groups):

* LEFT  — Validation Accuracy (%): each measured value is a filled marker
  (color = dataset, shape = encoder) with its 95% Wilson-CI whisker; the
  paper's *claimed* accuracy is a black open diamond; a thin light-gray
  tie line connects measured to claimed (the gap is the paper's argument).
* RIGHT — Validation Dice (%): same layout but **no CI whiskers** (Dice has
  no published CIs) — see the caption source comment below.

The claimed diamonds (and their tie lines) are omitted for the PanNuke rows,
which have no published comparison (Paper value NaN).

Data source: :func:`paper.figures.loaders.results_matrix` only — no value is
hard-coded.

Caption source note
-------------------
The Accuracy panel carries 95% Wilson-CI whiskers on every measured point;
the Dice panel deliberately has **no** whiskers because no published
confidence intervals exist for the macro-Dice claims. This asymmetry is
intentional and must be stated in the figure caption.

Output: ``paper/fig3_results_dotplot.pdf`` + ``paper/fig3_results_dotplot.png``
(via :func:`paper.figures.style.save_figure`).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import style
from .loaders import REPO_ROOT, results_matrix

# Apply the locked style contract once for this module.
style.apply()

# ---------------------------------------------------------------------------
# Experiment-group structure.
#
# The results matrix is *already* in group order (Groups 1-5). The group
# boundaries below are a fixed property of the experimental design (they do
# not depend on the data values). The short group labels are derived from the
# Phase / Run-Label structure of each group and kept short for the left
# margin.
# ---------------------------------------------------------------------------

#: (start_row, end_row_exclusive, short_label) in CSV row order.
GROUPS = [
    (0, 6, "Group 1: Baseline"),
    (6, 12, "Group 2: V2 Package"),
    (12, 16, "Group 3: PanNuke"),
    (16, 22, "Group 4: Isolation"),
    (22, 26, "Group 5: Ablation"),
]

#: Neutral gray for the 95% CI whisker (Acc panel only).
_CI_GRAY = "#888888"
#: Light gray for the measured<->claimed tie line.
_TIE_GRAY = "#cccccc"


def _short_label(row, run_num: int) -> str:
    """Build a short y-axis label like ``"01 TCGA VGG16"`` or
    ``"18 PANDA VGG16 +GN"`` from the row's config."""
    ds = str(row["Dataset"]).upper()
    enc = str(row["Encoder"]).upper().replace("_V2", "V2")
    parts = [f"{run_num:02d}", ds, enc]
    mods = []
    if bool(row["Use GradNorm"]):
        mods.append("+GN")
    if bool(row["Macenko"]):
        mods.append("+MAC")
    if not bool(row["Skip Connections"]):
        mods.append("no-skip")
    lr = str(row["Lambda Ratio (Seg:Cls)"])
    if lr and lr != "5:1":
        mods.append(f"\u03bb{lr}")
    if mods:
        parts.append(" ".join(mods))
    return " ".join(parts)


def _build_fig():
    """Build the Figure 3 ``Figure`` and return ``(fig, (ax_acc, ax_dice))``.

    Split out from :func:`build` so the layout can be inspected
    programmatically (see :func:`verify`).
    """
    matrix = results_matrix()
    n = len(matrix)

    # y positions: row 0 (run 1) at the top, row n-1 (run 26) at the bottom.
    y_pos = np.array([n - 1 - i for i in range(n)], dtype=float)

    fig, (ax_acc, ax_dice) = plt.subplots(
        1, 2, figsize=(7, 7.5), dpi=300, sharey=True
    )

    # --- Common y-axis (left panel carries the tick labels) --------------
    ax_acc.set_yticks(y_pos)
    ax_acc.set_yticklabels(
        [_short_label(matrix.iloc[i], i + 1) for i in range(n)], fontsize=7
    )
    ax_acc.set_ylim(-0.5, n - 0.5)
    ax_acc.set_xlim(0, 100)
    ax_acc.set_xlabel("Validation Accuracy (%)", fontweight="bold")
    ax_acc.set_title("Validation Accuracy (%)", fontweight="bold")

    ax_dice.set_xlim(0, 100)
    ax_dice.set_xlabel("Validation Dice (%)", fontweight="bold")
    ax_dice.set_title("Validation Dice (%)", fontweight="bold")
    # The right panel shares the y-axis; hide its (redundant) tick labels.
    ax_dice.yaxis.set_ticklabels([])

    # --- Group separators (thin horizontal lines) on both panels --------
    for (start, _end, _label) in GROUPS[1:]:
        # boundary between the previous group's last row and this group's first
        y_boundary = (y_pos[start - 1] + y_pos[start]) / 2.0
        ax_acc.axhline(y_boundary, color="black", linewidth=0.5,
                       linestyle="-", alpha=0.5, zorder=0)
        ax_dice.axhline(y_boundary, color="black", linewidth=0.5,
                        linestyle="-", alpha=0.5, zorder=0)

    # --- Per-row markers, CI whiskers, claimed diamonds, tie lines ------
    for i in range(n):
        row = matrix.iloc[i]
        y = y_pos[i]
        ds = str(row["Dataset"])
        enc = str(row["Encoder"])
        color = style.DATASET_COLORS[ds]
        marker = style.ENCODER_MARKERS[enc]

        meas_acc = float(row["Accuracy (%)"])
        meas_dice = float(row["Macro Dice (%)"])
        ci_lo = row["Acc 95% CI Lower"]
        ci_hi = row["Acc 95% CI Upper"]
        paper_acc = row["Paper Acc (%)"]
        paper_dice = row["Paper Dice (%)"]

        # --- Accuracy panel --------------------------------------------
        # 95% Wilson-CI whisker (horizontal, along the accuracy axis).
        if pd.notna(ci_lo) and pd.notna(ci_hi):
            flo, fhi = float(ci_lo), float(ci_hi)
            ax_acc.plot([flo, fhi], [y, y], color=_CI_GRAY, linewidth=1.0,
                        zorder=2, solid_capstyle="butt")
            cap = 0.15
            ax_acc.plot([flo, flo], [y - cap, y + cap], color=_CI_GRAY,
                        linewidth=1.0, zorder=2)
            ax_acc.plot([fhi, fhi], [y - cap, y + cap], color=_CI_GRAY,
                        linewidth=1.0, zorder=2)
        # Measured point (color = dataset, shape = encoder).
        ax_acc.plot(meas_acc, y, marker=marker, color=color,
                    markersize=6, linestyle="none", zorder=3)
        # Claimed diamond + tie line (skip if Paper value absent).
        if pd.notna(paper_acc):
            pa = float(paper_acc)
            ax_acc.plot([meas_acc, pa], [y, y], color=_TIE_GRAY,
                        linewidth=0.8, zorder=1)
            ax_acc.plot(pa, y, marker="D", mfc="none", mec="black",
                        markersize=6, linestyle="none", zorder=4)

        # --- Dice panel ------------------------------------------------
        # No CI whisker for Dice (no published CIs).
        ax_dice.plot(meas_dice, y, marker=marker, color=color,
                     markersize=6, linestyle="none", zorder=3)
        if pd.notna(paper_dice):
            pd_ = float(paper_dice)
            ax_dice.plot([meas_dice, pd_], [y, y], color=_TIE_GRAY,
                         linewidth=0.8, zorder=1)
            ax_dice.plot(pd_, y, marker="D", mfc="none", mec="black",
                         markersize=6, linestyle="none", zorder=4)

    # Subtle vertical grid on both panels.
    for ax in (ax_acc, ax_dice):
        ax.grid(axis="x", linestyle="--", alpha=style.GRID_ALPHA)

    # --- Layout, then place group labels in the left margin -------------
    fig.subplots_adjust(left=0.30, right=0.98, top=0.93, bottom=0.12)
    fig.canvas.draw()
    pos = ax_acc.get_position()
    ylim = ax_acc.get_ylim()
    for (start, end, label) in GROUPS:
        y_center = (y_pos[start] + y_pos[end - 1]) / 2.0
        y_frac = (y_center - ylim[0]) / (ylim[1] - ylim[0])
        y_fig = pos.y0 + y_frac * pos.height
        fig.text(pos.x0 - 0.015, y_fig, label, ha="right", va="center",
                 fontweight="bold", fontsize=7, color="black")

    # --- Shared legend (figure-level, lower center) --------------------
    handles = []
    for ds in ("TCGA", "PANDA", "SIIM", "PANNUKE"):
        handles.append(Patch(facecolor=style.DATASET_COLORS[ds],
                             edgecolor="black", linewidth=0.5, label=ds))
    for enc in ("vgg16", "mobilenet_v2"):
        handles.append(Line2D([], [], marker=style.ENCODER_MARKERS[enc],
                              linestyle="none", color="black", markersize=6,
                              label=enc))
    handles.append(Line2D([], [], marker="D", linestyle="none",
                          mfc="none", mec="black", markersize=6,
                          label="Claimed (paper)"))
    handles.append(Line2D([], [], color=_CI_GRAY, linewidth=1.0,
                          label="95% CI (Acc)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, 0.01))

    return fig, (ax_acc, ax_dice)


def build() -> list:
    """Render Figure 3 and save PDF + PNG. Returns the written paths."""
    fig, _ = _build_fig()
    out_dir = REPO_ROOT / "paper"
    return style.save_figure(fig, out_dir, "fig3_results_dotplot")


def verify() -> dict:
    """Verify the Figure 3 data contract (no visual read-back).

    Asserts:
      * exactly 26 rows;
      * every row has a parsed 95% CI (Acc Lower/Upper not NaN);
      * no NaN measured values (Accuracy / Macro Dice).
    Prints the row count. Returns a summary dict.
    """
    matrix = results_matrix()
    n = len(matrix)
    print(f"fig3 rows: {n}")
    assert n == 26, f"expected 26 rows, got {n}"

    # Every Acc row has a CI.
    n_ci = int((matrix["Acc 95% CI Lower"].notna()
                & matrix["Acc 95% CI Upper"].notna()).sum())
    assert n_ci == n, f"expected {n} rows with CI, got {n_ci}"

    # No NaN measured values.
    n_nan_acc = int(matrix["Accuracy (%)"].isna().sum())
    n_nan_dice = int(matrix["Macro Dice (%)"].isna().sum())
    assert n_nan_acc == 0, f"{n_nan_acc} NaN measured Accuracy values"
    assert n_nan_dice == 0, f"{n_nan_dice} NaN measured Dice values"

    # Build the figure to ensure it renders without error, then close it.
    fig, _ = _build_fig()
    plt.close(fig)

    return {
        "n_rows": n,
        "n_with_ci": n_ci,
        "n_nan_acc": n_nan_acc,
        "n_nan_dice": n_nan_dice,
    }


if __name__ == "__main__":
    for p in build():
        print(f"Saved: {p}")
