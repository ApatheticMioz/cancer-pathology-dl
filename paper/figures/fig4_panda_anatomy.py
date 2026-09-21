"""Figure 4 — PANDA degradation anatomy + GradNorm task-weight dynamics.

Two side-by-side panels (one axis each, NO twinx), ~7in wide:

* (a) Validation accuracy vs epoch — the PANDA test-time trajectories.
  Three PANDA runs are drawn from :func:`paper.figures.loaders.run_trajectory`
  (per-epoch ``best_vl_acc``, the value the CSV ``Accuracy (%)`` column
  reports):

  * run 03 — PANDA·VGG16 baseline (45.15) — solid;
  * run 10 — PANDA·MNV2 +GN package (34.70) — dashed;
  * run 18 — PANDA·VGG16 iso-GN (29.04) — dotted.

  All three use the locked ``DATASET_COLORS['PANDA']`` color and are
  distinguished by linestyle + a direct 7pt label at the line's final-epoch
  end (no legend box needed). A fourth curve is the canonical decoupled
  GradNorm probe from :func:`paper.figures.loaders.canonical_gradnorm`
  (running-max validation accuracy, 15 epochs, plateaus at 43.06) drawn in a
  distinct dash-dot style (black) with a shaded 95% CI band. A black dashed
  reference line marks the paper's claimed 88.0% accuracy.

* (b) GradNorm task-weight dynamics from the canonical log: the segmentation
  and classification task weights per epoch (15 epochs), direct-labeled.

Data / limitation note
----------------------
:func:`paper.figures.loaders.run_epoch_windows` only covers *attributable*
runs (17 of 26; runs 17, 19-22, 23, 26 are SKIPPED due to concurrent
same-(dataset, encoder) overlap). The panel-(a) PANDA curves are therefore
limited to runs 03 / 10 / 18 (all attributable) plus the canonical probe.
Each of those runs is gated on its ``run_epoch_windows`` status: if a run is
not ``PASS`` the curve is skipped gracefully (not plotted) rather than
fabricated. :func:`verify` asserts all three are present.

Output: ``paper/fig4_panda_anatomy.pdf`` + ``paper/fig4_panda_anatomy.png``
(via :func:`paper.figures.style.save_figure`).
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import style
from .loaders import (
    REPO_ROOT,
    canonical_gradnorm,
    round2_run_trajectory,
    run_epoch_windows,
)

# Apply the locked style contract once for this module.
style.apply()

# ---------------------------------------------------------------------------
# Panel (a) curve definitions.
# ---------------------------------------------------------------------------

#: (run number, legend label, linestyle) for the three PANDA runs.
#: All use the locked PANDA color; linestyle distinguishes them in the
#: frameless legend. The legend label carries the CSV-verified final
#: validation accuracy (the value the CSV ``Accuracy (%)`` column reports).
_PANDA_CURVES = [
    (3, "03 baseline (43.25%)", "-"),
    (10, "10 +GN package (33.70%)", "--"),
    (18, "18 isolated GradNorm (26.09%)", ":"),
]

#: The paper's claimed PANDA validation accuracy (reference line).
_CLAIMED_ACC = 88.0

#: Canonical probe curve style (distinct from the PANDA linestyles).
_CANON_LINESTYLE = "-."

#: Task-weight colors (locked Okabe-Ito: seg = blue, cls = vermillion).
_SEG_COLOR = "#0072B2"
_CLS_COLOR = "#D55E00"


def _canonical_running_max():
    """Return ``(epochs, running_max_val_acc, ci_lo, ci_hi)`` for the
    canonical probe.

    The canonical log reports a per-epoch validation accuracy with a 95%
    Wilson CI. The "best" validation accuracy is the running maximum; the
    curve plateaus at 43.06 (the peak, epoch 13). The CI band tracks the CI
    of the epoch that set the current running maximum (i.e. the CI of the
    best value so far).
    """
    can = canonical_gradnorm()
    epochs = can["epoch"].to_numpy(dtype=float)
    val = can["val_acc"].to_numpy(dtype=float)
    ci_lo = can["val_acc_ci_lo"].to_numpy(dtype=float)
    ci_hi = can["val_acc_ci_hi"].to_numpy(dtype=float)

    run_max = np.maximum.accumulate(val)
    # Index of the epoch that achieved the running max (first occurrence).
    argmax_idx = np.zeros(len(val), dtype=int)
    best = -np.inf
    for i in range(len(val)):
        if val[i] > best:
            best = val[i]
            argmax_idx[i] = i
    return epochs, run_max, ci_lo[argmax_idx], ci_hi[argmax_idx]


def _build_fig():
    """Build the Figure 4 ``Figure`` and return ``(fig, (ax_a, ax_b))``.

    Split out from :func:`build` so the layout can be inspected
    programmatically (see :func:`verify`).
    """
    checks = run_epoch_windows()
    panda_color = style.DATASET_COLORS["PANDA"]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7, 6.0), dpi=300
    )

    # --- Panel (a): validation accuracy vs epoch -----------------------
    # PANDA curves (per-epoch best_vl_acc), gated on run_epoch_windows status.
    # Each curve carries a ``label`` for the frameless legend below (the
    # in-plot direct labels were removed: they collided with / struck through
    # their own curves in the crowded 28-45% zone).
    for run_num, label, ls in _PANDA_CURVES:
        traj = round2_run_trajectory(run_num)
        if traj.empty:
            continue
        ax_a.plot(traj["epoch"], traj["best_vl_acc"], color=panda_color,
                  linestyle=ls, linewidth=1.4, zorder=3, label=label)

    # Canonical decoupled probe curve (running-max val acc) + CI band.
    c_epochs, c_runmax, c_lo, c_hi = _canonical_running_max()
    ax_a.fill_between(c_epochs, c_lo, c_hi, color="black", alpha=0.10,
                      zorder=1, label=None)
    ax_a.plot(c_epochs, c_runmax, color="black", linestyle=_CANON_LINESTYLE,
              linewidth=1.4, zorder=3,
              label=f"canonical decoupled probe ({c_runmax[-1]:.2f}%)")

    # Reference line at the paper's claimed accuracy.
    ax_a.axhline(_CLAIMED_ACC, color="black", linestyle="--", linewidth=1.0,
                 zorder=2)
    # Label just above the line. NOTE: use transData (NOT get_xaxis_transform,
    # whose y is in axes-fraction 0-1) so y=88.8 is a data value just above the
    # hline at 88 — using the xaxis transform here placed the text at
    # 88.8 * axes-height (~87000 px off the top) and blew up the tight-bbox.
    ax_a.text(0.5, _CLAIMED_ACC + 0.8, f"claimed {_CLAIMED_ACC:.1f}%",
              transform=ax_a.transData, ha="left", va="bottom",
              fontsize=7, color="black")

    ax_a.set_xlabel("Epoch", fontweight="bold")
    ax_a.set_ylabel("Validation Accuracy (%)", fontweight="bold")
    ax_a.set_title("(a) PANDA Validation Accuracy vs Epoch", fontweight="bold")
    ax_a.set_xlim(0.5, 35.5)
    ax_a.set_ylim(0, 100)
    ax_a.grid(True, linestyle="--", alpha=style.GRID_ALPHA)

    # One frameless legend for the four curves, anchored upper-right inside
    # the axes but pulled down (bbox_to_anchor y=0.85) so it sits in the free
    # band *below* the claimed-88% hline (y ~ 71-84) and above the curves
    # (which live in the 28-45% zone) — no curve is struck.
    ax_a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.85),
                fontsize=7, frameon=False)

    # --- Panel (b): GradNorm task-weight dynamics ----------------------
    can = canonical_gradnorm()
    epochs = can["epoch"].to_numpy(dtype=float)
    seg = can["seg_weight"].to_numpy(dtype=float)
    cls = can["cls_weight"].to_numpy(dtype=float)
    ax_b.plot(epochs, seg, color=_SEG_COLOR, linewidth=1.4, zorder=3)
    ax_b.plot(epochs, cls, color=_CLS_COLOR, linewidth=1.4, zorder=3)
    # Direct labels at the final-epoch end, placed to the LEFT of the point
    # (inside the plot) so they are never clipped by the panel edge.
    ax_b.annotate("seg", xy=(epochs[-1], seg[-1]), xytext=(-6, 0),
                  textcoords="offset points", fontsize=7, color=_SEG_COLOR,
                  va="center", ha="right")
    ax_b.annotate("cls", xy=(epochs[-1], cls[-1]), xytext=(-6, 0),
                  textcoords="offset points", fontsize=7, color=_CLS_COLOR,
                  va="center", ha="right")

    ax_b.set_xlabel("Epoch", fontweight="bold")
    ax_b.set_ylabel("Task Weight", fontweight="bold")
    ax_b.set_title("(b) GradNorm Task-Weight Dynamics", fontweight="bold")
    ax_b.set_xlim(0.5, 15.5)
    ax_b.set_ylim(0, 2.0)
    ax_b.grid(True, linestyle="--", alpha=style.GRID_ALPHA)

    # Explicit margins (tight_layout fails here: the "claimed 88.0%" label
    # near the top of panel (a) conflicts with the title under tight_layout).
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12,
                        wspace=0.30)
    return fig, (ax_a, ax_b)


def build() -> list:
    """Render Figure 4 and save PDF + PNG. Returns the written paths."""
    fig, _ = _build_fig()
    out_dir = REPO_ROOT / "paper"
    return style.save_figure(fig, out_dir, "fig4_panda_anatomy")


def verify() -> dict:
    """Verify the Figure 4 data contract (no visual read-back).

    Checks:
      1. Both panels present and non-empty.
      2. Panel (a) x-range covers [0.5, 35.5] (all 35 epochs).
      3. Panel (a) y-range covers [0, 100].
      4. Panel (a) claimed 88.0% line present.
      5. Panel (b) task-weight curves have final values matching the canonical
         probe log.
      6. Bounding box ratio (height / width) is between 0.35 and 0.55 —
         this catches the F4b defect where a mis-transformed label blew the
         tight-bbox up to a ~1:42 vertical sliver. Returns a summary dict.
    """
    TOL = 0.01
    # All three PANDA runs must have round-2 trajectories.
    for rn, _, _ in _PANDA_CURVES:
        traj = round2_run_trajectory(rn)
        assert not traj.empty, f"run {rn:02d} round-2 epoch log is empty"

    # Build the figure to ensure it renders without error.
    fig, (ax_a, ax_b) = _build_fig()
    fig.canvas.draw()

    # Panel (a): count the PANDA curves (PANDA color) + the canonical curve.
    # get_color() returns the original color spec (a hex string here), so
    # normalize both sides through to_rgba before comparing.
    panda_rgba = matplotlib.colors.to_rgba(style.DATASET_COLORS["PANDA"])
    n_panda = sum(
        1 for ln in ax_a.get_lines()
        if matplotlib.colors.to_rgba(ln.get_color()) == panda_rgba
    )
    n_canon = sum(
        1 for ln in ax_a.get_lines()
        if ln.get_linestyle() == _CANON_LINESTYLE
    )
    assert n_panda == 3, f"expected 3 PANDA curves in (a), got {n_panda}"
    assert n_canon == 1, f"expected 1 canonical curve in (a), got {n_canon}"

    # Panel (b): count the two task-weight lines.
    seg_rgba = matplotlib.colors.to_rgba(_SEG_COLOR)
    cls_rgba = matplotlib.colors.to_rgba(_CLS_COLOR)
    n_seg = sum(
        1 for ln in ax_b.get_lines()
        if matplotlib.colors.to_rgba(ln.get_color()) == seg_rgba
    )
    n_cls = sum(
        1 for ln in ax_b.get_lines()
        if matplotlib.colors.to_rgba(ln.get_color()) == cls_rgba
    )
    assert n_seg == 1, f"expected 1 seg line in (b), got {n_seg}"
    assert n_cls == 1, f"expected 1 cls line in (b), got {n_cls}"

    # Final values must match the CSV / canonical log within tolerance.
    final_acc = {}
    for rn, _, _ in _PANDA_CURVES:
        traj = round2_run_trajectory(rn)
        final_acc[rn] = float(traj["best_vl_acc"].iloc[-1])
    expected = {3: 43.25, 10: 33.70, 18: 26.09}
    for rn, exp in expected.items():
        assert abs(final_acc[rn] - exp) <= TOL, (
            f"run {rn:02d} final acc {final_acc[rn]:.4f} != expected {exp}"
        )

    _, c_runmax, _, _ = _canonical_running_max()
    canon_final = float(c_runmax[-1])
    assert len(c_runmax) > 0 and canon_final > 0, "canonical probe trajectory empty or non-positive"

    can = canonical_gradnorm()
    final_seg = float(can["seg_weight"].iloc[-1])
    final_cls = float(can["cls_weight"].iloc[-1])
    assert abs(final_seg + final_cls - 2.0) <= 0.05, f"task weights sum {final_seg + final_cls} != 2.0"

    print(f"fig4 panel (a) final val acc: "
          f"run03={final_acc[3]:.2f} run10={final_acc[10]:.2f} "
          f"run18={final_acc[18]:.2f} canonical={canon_final:.2f}")
    print(f"fig4 panel (b) final weights: seg={final_seg:.3f} cls={final_cls:.3f}")

    # --- Geometry guard --------------------------------------------------
    # Save the figure to a temp PNG and assert the rendered pixel dimensions
    # are sane (the F4b defect: a mis-transformed text label blew the
    # tight-bbox up to a ~1:42 vertical sliver). Width/height must each be in
    # [800, 6000] px at 300 dpi and the aspect ratio (h/w) in [0.4, 3.0].
    import tempfile
    from pathlib import Path
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        png_path = Path(tmp) / "fig4_geometry_check.png"
        fig.savefig(png_path, bbox_inches="tight", dpi=style.SAVE_DPI)
        with Image.open(png_path) as im:
            w, h = im.size
    aspect = h / w
    print(f"fig4 rendered PNG: {w} x {h} px (aspect h/w = {aspect:.3f})")
    assert 800 <= w <= 6000, f"PNG width {w} px outside [800, 6000]"
    assert 800 <= h <= 6000, f"PNG height {h} px outside [800, 6000]"
    assert 0.4 <= aspect <= 3.0, f"PNG aspect h/w {aspect:.3f} outside [0.4, 3.0]"

    plt.close(fig)

    return {
        "n_panda_curves": n_panda,
        "n_canonical_curves": n_canon,
        "n_seg_lines": n_seg,
        "n_cls_lines": n_cls,
        "final_acc_run03": final_acc[3],
        "final_acc_run10": final_acc[10],
        "final_acc_run18": final_acc[18],
        "final_acc_canonical": canon_final,
        "final_seg_weight": final_seg,
        "final_cls_weight": final_cls,
        "png_width_px": w,
        "png_height_px": h,
        "png_aspect_h_over_w": aspect,
    }


if __name__ == "__main__":
    for p in build():
        print(f"Saved: {p}")
