"""Figure 3 — 26-run claimed-vs-measured dot plot with 95% CIs (fold campaign).

The paper's centerpiece figure. Two side-by-side panels share a common
categorical y-axis of 26 rows (one row per run, in CSV order, which is
already grouped into the five experiment groups):

* LEFT  — Validation Accuracy (%): each measured value is a filled marker
  (color = dataset, shape = encoder) with its 95% across-folds CI whisker
  (k-fold CV mean ± z·(fold SD/√k)); the paper's *claimed* accuracy is a
  black open diamond; a thin light-gray tie line connects measured to
  claimed (the gap is the paper's argument).
* RIGHT — Validation Dice (%): same layout, with the 95% bootstrap
  across-folds CI whisker from ``dice_ci_summary.csv`` (the fold campaign
  now provides a Dice CI, so the panel is no longer CI-free).

The claimed diamonds (and their tie lines) are omitted for the PanNuke rows,
which have no published comparison (Paper value NaN).

Data source: the fold campaign — :func:`paper.figures.loaders.kfold_run_stats`
(k-fold CV accuracy mean/SD from ``kfold_*.json`` and the bootstrap
across-folds Dice CI from ``dice_ci_summary.csv``) joined to
:func:`paper.figures.loaders.results_matrix` for the claimed (paper) values.
No single-split number is hard-coded.

Caption source note
-------------------
The Accuracy panel carries 95% across-folds CI whiskers (k-fold mean ±
z·SD/√k); the Dice panel carries 95% bootstrap across-folds CI whiskers from
the fold campaign. Both panels now show CIs.

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
from .loaders import REPO_ROOT, kfold_run_stats, results_matrix

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


#: Encoder -> short display name (kept short so row labels fit the margin).
_ENCODER_SHORT = {"vgg16": "VGG16", "mobilenet_v2": "MNV2"}

#: Explicit per-run row-label modifiers (run number -> modifier string, or
#: ``None`` for the plain baseline). The base part of each label is derived
#: from the row's Dataset/Encoder columns; only the differentiating modifier
#: is pinned here so it is never lost to length-based truncation (the F3c
#: defect: greedy truncation kept the redundant ``no-Mac`` on v1 baselines
#: while dropping the differentiating ``λ`` ratios on runs 19-22 and the
#: ``no-Mac``/``no-skip`` on runs 23-26, making rows 23 and 26 identical).
_LABEL_MODIFIERS = {
    1: None, 2: None, 3: None, 4: None, 5: None, 6: None,
    7: "+GN", 8: "+GN", 9: "+GN", 10: "+GN", 11: "+GN", 12: "+GN",
    13: None, 14: None, 15: "+GN", 16: "+GN",
    17: "\u03b71e-4", 18: "iso-GN",
    19: "\u03bb1:1", 20: "\u03bb5:1", 21: "\u03bb1:10", 22: "\u03bb10:1",
    23: "no-Mac", 24: "no-Mac", 25: "no-skip", 26: "no-skip",
}

#: Expected ``(Dataset, Encoder)`` per run number — a sanity guard so the
#: label mapping cannot silently desync from the data.
_RUN_DATASET_ENCODER = {
    1: ("TCGA", "vgg16"), 2: ("TCGA", "mobilenet_v2"),
    3: ("PANDA", "vgg16"), 4: ("PANDA", "mobilenet_v2"),
    5: ("SIIM", "vgg16"), 6: ("SIIM", "mobilenet_v2"),
    7: ("TCGA", "vgg16"), 8: ("TCGA", "mobilenet_v2"),
    9: ("PANDA", "vgg16"), 10: ("PANDA", "mobilenet_v2"),
    11: ("SIIM", "vgg16"), 12: ("SIIM", "mobilenet_v2"),
    13: ("PANNUKE", "vgg16"), 14: ("PANNUKE", "mobilenet_v2"),
    15: ("PANNUKE", "vgg16"), 16: ("PANNUKE", "mobilenet_v2"),
    17: ("PANDA", "vgg16"), 18: ("PANDA", "vgg16"),
    19: ("PANDA", "vgg16"), 20: ("PANDA", "vgg16"),
    21: ("PANDA", "vgg16"), 22: ("PANDA", "vgg16"),
    23: ("PANDA", "mobilenet_v2"), 24: ("PANNUKE", "mobilenet_v2"),
    25: ("TCGA", "mobilenet_v2"), 26: ("PANDA", "mobilenet_v2"),
}


def _short_label(row, run_num: int) -> str:
    """Build the per-run y-axis label like ``"01 TCGA·VGG16"`` or
    ``"19 PANDA·VGG16 λ1:1"``.

    The base part (``<run#> <DATASET>·<ENCODER>``) is derived from the row's
    Dataset/Encoder columns; the differentiating modifier is pinned by
    :data:`_LABEL_MODIFIERS` (never lost to length-based truncation). A
    sanity guard asserts the row's dataset+encoder match the expected value
    for that run number so the mapping cannot silently desync from the data.
    """
    ds = str(row["Dataset"]).upper()
    enc = _ENCODER_SHORT.get(str(row["Encoder"]), str(row["Encoder"]).upper())
    expected_ds, expected_enc = _RUN_DATASET_ENCODER[run_num]
    if ds != expected_ds or str(row["Encoder"]) != expected_enc:
        raise AssertionError(
            f"run {run_num:02d} dataset/encoder mismatch: "
            f"got {ds}/{row['Encoder']}, expected {expected_ds}/{expected_enc}"
        )
    label = f"{run_num:02d} {ds}\u00b7{enc}"
    modifier = _LABEL_MODIFIERS[run_num]
    if modifier is not None:
        label = f"{label} {modifier}"
    return label


def _build_fig():
    """Build the Figure 3 ``Figure`` and return ``(fig, (ax_acc, ax_dice))``.

    Split out from :func:`build` so the layout can be inspected
    programmatically (see :func:`verify`).
    """
    matrix = results_matrix()
    n = len(matrix)

    # y positions: row 0 (run 1) at the top, row n-1 (run 26) at the bottom.
    y_pos = np.array([n - 1 - i for i in range(n)], dtype=float)

    # The two panels share a single y-axis of 26 rows (sharey=True). The
    # per-run row labels are set on that shared axis; matplotlib draws them
    # only on the leftmost (Accuracy) panel, so the right panel's left edge
    # shows tick marks but no duplicate labels (no collision in the
    # inter-panel gap). Both panels therefore carry 26 y-tick positions.
    #
    # NOTE: we must NOT call set_ticklabels([]) on the shared axis — that
    # would clear the labels for BOTH panels (the original F3 bug that left
    # the y-axis bare).
    fig, (ax_acc, ax_dice) = plt.subplots(
        1, 2, figsize=(7, 7.5), dpi=300, sharey=True
    )

    # --- Per-run row labels on the shared y-axis -----------------------
    row_labels = [_short_label(matrix.iloc[i], i + 1) for i in range(n)]
    ax_acc.set_yticks(y_pos)
    ax_acc.set_yticklabels(row_labels, fontsize=7)
    ax_acc.set_ylim(-0.5, n - 0.5)
    # x headroom so markers / CI whiskers / claimed diamonds never clip.
    ax_acc.set_xlim(-2, 105)
    ax_acc.set_xticks([0, 20, 40, 60, 80, 100])
    ax_acc.set_xlabel("Validation Accuracy (%)", fontweight="bold")
    ax_acc.set_title("Validation Accuracy (%)", fontweight="bold")

    ax_dice.set_ylim(-0.5, n - 0.5)
    ax_dice.set_xlim(-2, 105)
    ax_dice.set_xticks([0, 20, 40, 60, 80, 100])
    ax_dice.set_xlabel("Validation Dice (%)", fontweight="bold")
    ax_dice.set_title("Validation Dice (%)", fontweight="bold")

    # --- Group separators (thin horizontal lines) on both panels --------
    for (start, _end, _label) in GROUPS[1:]:
        # boundary between the previous group's last row and this group's first
        y_boundary = (y_pos[start - 1] + y_pos[start]) / 2.0
        ax_acc.axhline(y_boundary, color="black", linewidth=0.5,
                       linestyle="-", alpha=0.5, zorder=0)
        ax_dice.axhline(y_boundary, color="black", linewidth=0.5,
                        linestyle="-", alpha=0.5, zorder=0)

    # --- Per-row markers, CI whiskers, claimed diamonds, tie lines ------
    # Measured values + CIs come from the fold campaign (k-fold CV accuracy
    # mean/SD and the bootstrap across-folds Dice CI); the claimed (paper)
    # values come from the results matrix. No single-split number is used.
    for i in range(n):
        row = matrix.iloc[i]
        y = y_pos[i]
        ds = str(row["Dataset"])
        enc = str(row["Encoder"])
        color = style.DATASET_COLORS[ds]
        marker = style.ENCODER_MARKERS[enc]

        stats = kfold_run_stats(i + 1)
        if stats is None:
            # Not part of the fold campaign: fall back to the CSV row.
            meas_acc = float(row["Accuracy (%)"])
            meas_dice = float(row["Macro Dice (%)"])
            acc_lo = float(row["Acc 95% CI Lower"])
            acc_hi = float(row["Acc 95% CI Upper"])
            dice_lo = float(row["Dice 95% CI Lower (bootstrap)"])
            dice_hi = float(row["Dice 95% CI Upper (bootstrap)"])
        else:
            meas_acc = stats["acc_point"]
            meas_dice = stats["dice_point"]
            acc_lo = stats["acc_ci_lo"]
            acc_hi = stats["acc_ci_hi"]
            dice_lo = stats["dice_ci_lo"]
            dice_hi = stats["dice_ci_hi"]
        paper_acc = row["Paper Acc (%)"]
        paper_dice = row["Paper Dice (%)"]

        # --- Accuracy panel --------------------------------------------
        # 95% across-folds CI whisker (horizontal, along the accuracy axis).
        if pd.notna(acc_lo) and pd.notna(acc_hi):
            flo, fhi = acc_lo, acc_hi
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
        # 95% bootstrap across-folds CI whisker (the fold campaign provides
        # a Dice CI, so the panel is no longer CI-free).
        if pd.notna(dice_lo) and pd.notna(dice_hi):
            dlo, dhi = dice_lo, dice_hi
            ax_dice.plot([dlo, dhi], [y, y], color=_CI_GRAY, linewidth=1.0,
                         zorder=2, solid_capstyle="butt")
            cap = 0.15
            ax_dice.plot([dlo, dlo], [y - cap, y + cap], color=_CI_GRAY,
                         linewidth=1.0, zorder=2)
            ax_dice.plot([dhi, dhi], [y - cap, y + cap], color=_CI_GRAY,
                         linewidth=1.0, zorder=2)
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
    # The left margin is widened so the per-run row labels fit, and the
    # group labels are placed to the *left* of the row labels (no overlap).
    fig.subplots_adjust(left=0.42, right=0.98, top=0.93, bottom=0.12)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pos = ax_acc.get_position()
    ylim = ax_acc.get_ylim()
    # Leftmost edge of the row tick labels (in figure-fraction coords).
    row_left = min(
        t.get_window_extent(renderer).x0 / fig.get_figwidth() / fig.dpi
        for t in ax_acc.get_yticklabels()
    )
    for (start, end, label) in GROUPS:
        y_center = (y_pos[start] + y_pos[end - 1]) / 2.0
        y_frac = (y_center - ylim[0]) / (ylim[1] - ylim[0])
        y_fig = pos.y0 + y_frac * pos.height
        fig.text(row_left - 0.015, y_fig, label, ha="right", va="center",
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
                          label="95% CI (across-folds)"))
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
      * no NaN measured values (Accuracy / Macro Dice);
      * every row has a non-empty per-run tick label (26 per panel);
      * all 26 rendered labels match the explicit run-number mapping exactly
        (and rows 23 vs 26 are no longer identical).
    Prints the row count, the rendered limits / tick-label counts, and the
    rendered labels. Returns a summary dict.
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

    # Every row must be covered by the fold campaign (k-fold summary +
    # across-folds bootstrap Dice CI) so the plotted points are the
    # fold-aware values, not single-split numbers.
    n_fold = sum(1 for i in range(n) if kfold_run_stats(i + 1) is not None)
    assert n_fold == n, f"expected {n} fold-campaign rows, got {n_fold}"

    # Build the figure to ensure it renders without error.
    fig, (ax_acc, ax_dice) = _build_fig()
    fig.canvas.draw()

    # Both panels share one y-axis of 26 rows, so each panel must expose 26
    # y-tick positions, and the shared axis must carry 26 non-empty per-run
    # text labels (the original F3 bug left these empty).
    n_acc_ticks = len(ax_acc.get_yticks())
    n_dice_ticks = len(ax_dice.get_yticks())
    shared_labels = [t.get_text() for t in ax_acc.get_yticklabels()]
    n_nonempty = sum(1 for t in shared_labels if t.strip())
    assert n_acc_ticks == n, f"expected {n} acc y-ticks, got {n_acc_ticks}"
    assert n_dice_ticks == n, f"expected {n} dice y-ticks, got {n_dice_ticks}"
    assert n_nonempty == n, (
        f"expected {n} non-empty row labels, got {n_nonempty}"
    )

    # Every rendered label must match the explicit run-number mapping exactly
    # (the F3c fix: no greedy truncation, so no label is silently altered).
    expected_labels = [_short_label(matrix.iloc[i], i + 1) for i in range(n)]
    mismatches = [
        (i + 1, got, exp)
        for i, (got, exp) in enumerate(zip(shared_labels, expected_labels))
        if got != exp
    ]
    assert not mismatches, (
        f"{len(mismatches)} row label(s) do not match the explicit mapping: "
        f"{mismatches[:3]}"
    )
    # The F3c defect specifically: rows 23 and 26 must no longer be identical.
    assert shared_labels[22] != shared_labels[25], (
        "rows 23 and 26 render identical labels (F3c defect not fixed)"
    )

    acc_xlim = tuple(float(v) for v in ax_acc.get_xlim())
    acc_ylim = tuple(float(v) for v in ax_acc.get_ylim())
    dice_xlim = tuple(float(v) for v in ax_dice.get_xlim())
    print(f"fig3 acc xlim={acc_xlim} ylim={acc_ylim} "
          f"y-ticks={n_acc_ticks} (non-empty labels {n_nonempty})")
    print(f"fig3 dice xlim={dice_xlim} y-ticks={n_dice_ticks}")
    print("fig3 rendered row labels:")
    for lbl in shared_labels:
        print(f"    {lbl}")

    plt.close(fig)

    return {
        "n_rows": n,
        "n_with_ci": n_ci,
        "n_fold_campaign_rows": n_fold,
        "n_nan_acc": n_nan_acc,
        "n_nan_dice": n_nan_dice,
        "n_acc_yticks": n_acc_ticks,
        "n_dice_yticks": n_dice_ticks,
        "n_nonempty_row_labels": n_nonempty,
        "n_labels_match_mapping": n - len(mismatches),
        "acc_xlim": acc_xlim,
        "acc_ylim": acc_ylim,
        "dice_xlim": dice_xlim,
    }


if __name__ == "__main__":
    for p in build():
        print(f"Saved: {p}")
