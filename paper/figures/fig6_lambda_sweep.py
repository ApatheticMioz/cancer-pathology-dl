"""Figure 6 — Static loss-weight (lambda_seg:lambda_cls) ratio sweep.

Two side-by-side panels sharing a categorical x-axis of the four
``lambda_seg:lambda_cls`` ratios (one axis each, NO twinx), ~7in x ~3.2in:

* (a) Validation Accuracy (%) — each sweep run is a marker
  (``DATASET_COLORS['PANDA']`` + ``vgg16`` circle) with its 95% across-folds
  CI whisker (k-fold CV mean ± z·(fold SD/√k)); a black dashed hline marks
  the paper's *claimed* accuracy of the run-03 baseline (read from the CSV
  ``Paper Acc (%)`` column).
* (b) Validation Dice (%) — the **per-epoch validation macro-Dice**
  (``mean_val_dice × 100`` from the ``kfold_<run>.json`` summary) as
  points only (no whiskers); the claimed hline marks the run-03
  ``Paper Dice (%)`` (98.0 %).

The four sweep runs are the PANDA·VGG16 Group-4 lambda-sweep rows (raw
input, eta=1e-3, no GradNorm): 1:1, 5:1, 1:10, 10:1. They are matched by the
CSV ``Lambda Ratio (Seg:Cls)`` column (never hard-coded) and drawn in the
fixed display order ``[1:10, 1:1, 5:1, 10:1]``.

Data source: the fold campaign — :func:`paper.figures.loaders.kfold_run_stats`
(k-fold CV mean accuracy + bootstrap across-folds Dice) joined to
:func:`paper.figures.loaders.results_matrix` for the claimed (paper) values.
No single-split number is hard-coded.

Output: ``paper/fig6_lambda_sweep.pdf`` + ``paper/fig6_lambda_sweep.png``
(via :func:`paper.figures.style.save_figure`).
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import style
from .loaders import (
    REPO_ROOT,
    kfold_run_stats,
    kfold_summary,
    results_matrix,
    run_name_for_run_num,
)

# Apply the locked style contract once for this module.
style.apply()

# ---------------------------------------------------------------------------
# Sweep structure.
# ---------------------------------------------------------------------------

#: Fixed display order of the lambda_seg:lambda_cls ratios (categorical x).
_LAMBDA_ORDER = ["1:10", "1:1", "5:1", "10:1"]

#: Group-4 (Isolation) row slice in CSV order (0-indexed, exclusive end).
_GROUP4_SLICE = (16, 22)
#: Group-1 (Baseline) row slice in CSV order (0-indexed, exclusive end).
_GROUP1_SLICE = (0, 6)

#: Neutral gray for the 95% CI whisker (Acc panel only).
_CI_GRAY = "#888888"
#: Light gray for the thin connecting line.
_TIE_GRAY = "#cccccc"


def _sweep_rows():
    """Return the four PANDA·VGG16 Group-4 lambda-sweep rows in
    :data:`_LAMBDA_ORDER`.

    Matched by the CSV ``Lambda Ratio (Seg:Cls)`` column (never hard-coded):
    the Group-4 slice filtered to ``Dataset==PANDA``, ``Encoder==vgg16``,
    ``LR==0.001`` (raw input, eta=1e-3), ``Use GradNorm==False``. Each of the
    four ratios must appear exactly once.
    """
    matrix = results_matrix()
    g4 = matrix.iloc[_GROUP4_SLICE[0]:_GROUP4_SLICE[1]]
    sel = g4[
        (g4["Dataset"] == "PANDA")
        & (g4["Encoder"] == "vgg16")
        & (g4["LR"] == 0.001)
        & (g4["Use GradNorm"] == False)  # noqa: E712
    ]
    by_ratio = {str(r["Lambda Ratio (Seg:Cls)"]): r for _, r in sel.iterrows()}
    rows = []
    for ratio in _LAMBDA_ORDER:
        if ratio not in by_ratio:
            raise AssertionError(f"lambda ratio {ratio!r} not found in Group-4 sweep")
        rows.append(by_ratio[ratio])
    return rows


def _claimed_row():
    """Return the run-03 PANDA·VGG16 baseline row (Group-1) whose
    ``Paper Acc (%)`` / ``Paper Dice (%)`` define the claimed hlines."""
    matrix = results_matrix()
    g1 = matrix.iloc[_GROUP1_SLICE[0]:_GROUP1_SLICE[1]]
    sel = g1[(g1["Dataset"] == "PANDA") & (g1["Encoder"] == "vgg16")]
    if len(sel) != 1:
        raise AssertionError(f"expected 1 run-03 PANDA/vgg16 row, got {len(sel)}")
    return sel.iloc[0]


def _build_fig():
    """Build the Figure 6 ``Figure`` and return ``(fig, (ax_a, ax_b))``.

    Split out from :func:`build` so the layout can be inspected
    programmatically (see :func:`verify`).
    """
    rows = _sweep_rows()
    claimed = _claimed_row()
    claimed_acc = float(claimed["Paper Acc (%)"])
    claimed_dice = float(claimed["Paper Dice (%)"])

    panda_color = style.DATASET_COLORS["PANDA"]
    marker = style.ENCODER_MARKERS["vgg16"]

    xs = np.arange(len(_LAMBDA_ORDER), dtype=float)
    # Fold-campaign values: the across-folds k-fold CV mean accuracy (point)
    # and its 95% CI (mean ± z·(fold SD/√k)), plus the bootstrap
    # across-folds Dice point estimate. No single-split number is used.
    # Each sweep row is a Group-4 results-matrix row; its 1-based run number
    # is its position in the matrix (the CSV is in run order), so ``r.name``
    # (the preserved index) + 1 is the run number.
    run_nums = [int(r.name) + 1 for r in rows]
    stats = [kfold_run_stats(rn) for rn in run_nums]
    acc = np.array([s["acc_point"] for s in stats])
    ci_lo = np.array([s["acc_ci_lo"] for s in stats])
    ci_hi = np.array([s["acc_ci_hi"] for s in stats])
    # Panel (b) estimator: the per-epoch validation macro-Dice
    # (``mean_val_dice × 100`` from the kfold summary), NOT the pooled
    # empty-credit Dice floor from the bootstrap CI table.
    dice = np.array(
        [
            float(kfold_summary(run_name_for_run_num(rn))["mean_val_dice"]) * 100.0
            for rn in run_nums
        ]
    )

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7, 3.2), dpi=300, sharex=True
    )

    # --- Panel (a): Accuracy --------------------------------------------
    # Thin light connecting line (drawn first, behind the markers).
    ax_a.plot(xs, acc, color=_TIE_GRAY, linewidth=0.8, zorder=1)
    # 95% across-folds CI whiskers (vertical, along the accuracy axis).
    cap = 0.15
    for x, lo, hi in zip(xs, ci_lo, ci_hi):
        ax_a.plot([x, x], [lo, hi], color=_CI_GRAY, linewidth=1.0,
                  zorder=2, solid_capstyle="butt")
        ax_a.plot([x - cap, x + cap], [lo, lo], color=_CI_GRAY, linewidth=1.0,
                  zorder=2)
        ax_a.plot([x - cap, x + cap], [hi, hi], color=_CI_GRAY, linewidth=1.0,
                  zorder=2)
    # Measured markers (PANDA color, vgg16 circle).
    for x, y in zip(xs, acc):
        ax_a.plot(x, y, marker=marker, color=panda_color, markersize=6,
                  linestyle="none", zorder=3)
    # 7pt value labels near each point.
    for x, y in zip(xs, acc):
        ax_a.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 8),
                      textcoords="offset points", fontsize=7,
                      ha="center", va="bottom", color="black")
    # Claimed hline (run-03 Paper Acc) + label just below the line.
    ax_a.axhline(claimed_acc, color="black", linestyle="--", linewidth=1.0,
                 zorder=2)
    ax_a.text(0.2, claimed_acc - 2.0, f"claimed {claimed_acc:.1f}%",
              transform=ax_a.transData, ha="left", va="top",
              fontsize=7, color="black")

    # --- Panel (b): Dice (per-epoch validation macro-Dice) -------------
    # Points only — no CI whiskers (the per-epoch macro-Dice is a point
    # estimate from the kfold summary, not a pooled bootstrap floor).
    # Thin light connecting line.
    ax_b.plot(xs, dice, color=_TIE_GRAY, linewidth=0.8, zorder=1)
    # Measured markers.
    for x, y in zip(xs, dice):
        ax_b.plot(x, y, marker=marker, color=panda_color, markersize=6,
                  linestyle="none", zorder=3)
    # 7pt value labels near each point (placed below points to avoid colliding with claimed hline).
    for x, y in zip(xs, dice):
        ax_b.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, -11),
                      textcoords="offset points", fontsize=7,
                      ha="center", va="top", color="black")
    # Claimed hline (run-03 Paper Dice) + label just above the line.
    ax_b.axhline(claimed_dice, color="black", linestyle="--", linewidth=1.0,
                 zorder=2)
    ax_b.text(0.2, claimed_dice + 1.5, f"claimed {claimed_dice:.1f}%",
              transform=ax_b.transData, ha="left", va="bottom",
              fontsize=7, color="black")

    # --- Shared categorical x-axis --------------------------------------
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(_LAMBDA_ORDER)
    ax_a.set_xlim(-0.5, len(_LAMBDA_ORDER) - 0.5)
    ax_a.set_xlabel("\u03bb_seg : \u03bb_cls", fontweight="bold")
    ax_a.set_ylim(0, 108)
    ax_a.set_ylabel("Validation Accuracy (%)", fontweight="bold")
    ax_a.set_title("(a) Validation Accuracy (%)", fontweight="bold")
    ax_a.grid(axis="y", linestyle="--", alpha=style.GRID_ALPHA)

    ax_b.set_ylim(0, 108)
    ax_b.set_ylabel("Validation Dice (%)", fontweight="bold")
    ax_b.set_title("(b) Validation Dice (%)", fontweight="bold")
    ax_b.grid(axis="y", linestyle="--", alpha=style.GRID_ALPHA)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18,
                        wspace=0.30)
    return fig, (ax_a, ax_b)


def build() -> list:
    """Render Figure 6 and save PDF + PNG. Returns the written paths."""
    fig, _ = _build_fig()
    out_dir = REPO_ROOT / "paper"
    return style.save_figure(fig, out_dir, "fig6_lambda_sweep")


def verify() -> dict:
    """Verify the Figure 6 data contract (no visual read-back).

    Asserts:
      * exactly 4 points per panel (PANDA color, vgg16 marker);
      * each panel's point values match the fold-campaign values (k-fold CV
        mean accuracy / per-epoch validation macro-Dice) within tolerance
        0.01;
      * the claimed hlines equal the CSV ``Paper Acc (%)`` / ``Paper Dice (%)``
        of the run-03 baseline;
      * the rendered "claimed XX%" labels match the CSV values;
      * geometry guard: the rendered PNG is a sane size (width/height in
        [800, 6000] px at 300 dpi, aspect h/w in [0.4, 3.0]).
    Returns a summary dict.
    """
    TOL = 0.01
    rows = _sweep_rows()
    claimed = _claimed_row()
    claimed_acc = float(claimed["Paper Acc (%)"])
    claimed_dice = float(claimed["Paper Dice (%)"])
    # Fold-campaign expected values (k-fold CV mean acc + per-epoch
    # validation macro-Dice from the kfold summary).
    run_nums = [int(r.name) + 1 for r in rows]
    stats = [kfold_run_stats(rn) for rn in run_nums]
    acc_csv = [s["acc_point"] for s in stats]
    dice_csv = [
        float(kfold_summary(run_name_for_run_num(rn))["mean_val_dice"]) * 100.0
        for rn in run_nums
    ]

    fig, (ax_a, ax_b) = _build_fig()
    fig.canvas.draw()

    panda_rgba = matplotlib.colors.to_rgba(style.DATASET_COLORS["PANDA"])

    def _data_points(ax):
        """Return the (x, y) of the PANDA/vgg16 markers on ``ax``, sorted by x."""
        pts = []
        for ln in ax.get_lines():
            if ln.get_marker() != "None" and \
                    matplotlib.colors.to_rgba(ln.get_color()) == panda_rgba:
                x = float(ln.get_xdata()[0])
                y = float(ln.get_ydata()[0])
                pts.append((x, y))
        pts.sort()
        return pts

    pts_a = _data_points(ax_a)
    pts_b = _data_points(ax_b)
    assert len(pts_a) == 4, f"expected 4 points in (a), got {len(pts_a)}"
    assert len(pts_b) == 4, f"expected 4 points in (b), got {len(pts_b)}"

    # Values must match the CSV within tolerance.
    for (x, y), exp in zip(pts_a, acc_csv):
        assert abs(y - exp) <= TOL, f"acc point {y:.4f} != CSV {exp}"
    for (x, y), exp in zip(pts_b, dice_csv):
        assert abs(y - exp) <= TOL, f"dice point {y:.4f} != CSV {exp}"

    # Claimed hlines: find the black dashed horizontal line in each panel.
    def _claimed_hline(ax):
        for ln in ax.get_lines():
            if ln.get_linestyle() == "--" and \
                    matplotlib.colors.to_rgba(ln.get_color()) == \
                    matplotlib.colors.to_rgba("black"):
                ydata = ln.get_ydata()
                if len(ydata) >= 2 and \
                        all(abs(v - ydata[0]) < 1e-9 for v in ydata):
                    return float(ydata[0])
        return None

    hline_a = _claimed_hline(ax_a)
    hline_b = _claimed_hline(ax_b)
    assert hline_a is not None, "no claimed hline found in (a)"
    assert hline_b is not None, "no claimed hline found in (b)"
    assert abs(hline_a - claimed_acc) <= TOL, \
        f"claimed acc hline {hline_a} != CSV {claimed_acc}"
    assert abs(hline_b - claimed_dice) <= TOL, \
        f"claimed dice hline {hline_b} != CSV {claimed_dice}"

    # The rendered "claimed XX%" labels must match the CSV values.
    texts = {t.get_text() for t in ax_a.texts} | {t.get_text() for t in ax_b.texts}
    assert f"claimed {claimed_acc:.1f}%" in texts, \
        f"claimed acc label {claimed_acc:.1f}% not rendered"
    assert f"claimed {claimed_dice:.1f}%" in texts, \
        f"claimed dice label {claimed_dice:.1f}% not rendered"

    print(f"fig6 claimed (run-03): acc={claimed_acc:.1f} dice={claimed_dice:.1f}")
    print(f"fig6 acc points:  {[f'{y:.2f}' for _, y in pts_a]}")
    print(f"fig6 dice points: {[f'{y:.2f}' for _, y in pts_b]}")

    # --- Geometry guard --------------------------------------------------
    # Save the figure to a temp PNG and assert the rendered pixel dimensions
    # are sane (catches a mis-transformed label blowing up the tight-bbox).
    import tempfile
    from pathlib import Path
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        png_path = Path(tmp) / "fig6_geometry_check.png"
        fig.savefig(png_path, bbox_inches="tight", dpi=style.SAVE_DPI)
        with Image.open(png_path) as im:
            w, h = im.size
    aspect = h / w
    print(f"fig6 rendered PNG: {w} x {h} px (aspect h/w = {aspect:.3f})")
    assert 800 <= w <= 6000, f"PNG width {w} px outside [800, 6000]"
    assert 800 <= h <= 6000, f"PNG height {h} px outside [800, 6000]"
    assert 0.4 <= aspect <= 3.0, f"PNG aspect h/w {aspect:.3f} outside [0.4, 3.0]"

    plt.close(fig)

    return {
        "n_points_acc": len(pts_a),
        "n_points_dice": len(pts_b),
        "claimed_acc": claimed_acc,
        "claimed_dice": claimed_dice,
        "acc_points": [y for _, y in pts_a],
        "dice_points": [y for _, y in pts_b],
        "png_width_px": w,
        "png_height_px": h,
        "png_aspect_h_over_w": aspect,
    }


if __name__ == "__main__":
    for p in build():
        print(f"Saved: {p}")
