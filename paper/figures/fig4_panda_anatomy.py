"""Figure 4 — PANDA degradation anatomy + GradNorm task-weight dynamics.

Two side-by-side panels (one axis each, NO twinx), ~7in wide:

* (a) Validation accuracy vs epoch — **fold-1 training trajectories from the
  fold campaign** (``results/round2/kfold_<cfg>_fold1of5/epoch_log.jsonl``,
  per-epoch ``best_vl_acc``):

  * 03 — PANDA·VGG16 baseline (``g1_panda_vgg16``) — solid;
  * 10 — PANDA·MNV2 +GN package (``g2_panda_mobilenet_v2``) — dashed;
  * 18 — PANDA·VGG16 iso-GN (``g4_panda_isolate_gn``) — dotted.

  Each curve's legend label carries the **five-fold mean** accuracy from the
  ``kfold_<cfg>.json`` summary (34.51 / 37.56 / 28.21) — read from the
  summary, never hard-coded. A fourth curve is the canonical decoupled
  GradNorm probe's fold-1 trajectory
  (``kfold_canonical_gradnorm_fold1of5``, 11 epochs) in a distinct dash-dot
  style (black), with a horizontal reference line at the five-fold mean
  (22.70) and a horizontal shaded 95% fold-bootstrap CI band
  [14.66, 30.73] (from :func:`paper.figures.loaders.kfold_acc_ci`).

* (b) GradNorm task-weight dynamics from the **seeded** canonical probe log
  (``results/round2/canonical_gradnorm_probe/probe_log.jsonl``, seed 42):
  the segmentation and classification task weights per epoch,
  direct-labeled. The panel title discloses the seeded-probe provenance.

Data / limitation note
----------------------
Panel (a) plots the fold-1 epoch logs of the fold campaign (the same
provenance as the k-fold summaries in the results matrix), not the legacy
single-run trajectories. The canonical curve is the fold-1 trajectory of
the seeded canonical GradNorm run; its reference line and CI band are the
across-folds mean and 95% fold-bootstrap CI of the same run.

Output: ``paper/fig4_panda_anatomy.pdf`` + ``paper/fig4_panda_anatomy.png``
(via :func:`paper.figures.style.save_figure`).
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from . import style
from .loaders import (
    REPO_ROOT,
    canonical_gradnorm,
    kfold_acc_ci,
    kfold_summary,
    per_run_epoch_log,
)

# Apply the locked style contract once for this module.
style.apply()

# ---------------------------------------------------------------------------
# Panel (a) curve definitions.
# ---------------------------------------------------------------------------

#: (kfold run name, short name, linestyle) for the three PANDA runs.
#: All use the locked PANDA color; linestyle distinguishes them in the
#: frameless legend. The legend label is built at render time from the
#: kfold summary's five-fold mean (never hard-coded) — see
#: :func:`_panda_label`.
_PANDA_CURVES = [
    ("g1_panda_vgg16", "03 baseline", "-"),
    ("g2_panda_mobilenet_v2", "10 +GN package", "--"),
    ("g4_panda_isolate_gn", "18 isolated GradNorm", ":"),
]

#: The canonical decoupled GradNorm run (fold campaign).
_CANON_RUN = "canonical_gradnorm"

#: Canonical probe curve style (distinct from the PANDA linestyles).
_CANON_LINESTYLE = "-."

#: Task-weight colors (locked Okabe-Ito: seg = blue, cls = vermillion).
_SEG_COLOR = "#0072B2"
_CLS_COLOR = "#D55E00"


def _fold1_trajectory(run_name: str) -> pd.DataFrame:
    """Per-epoch fold-1 trajectory for one fold-campaign run.

    Reads ``results/round2/kfold_<run_name>_fold1of5/epoch_log.jsonl`` and
    returns a DataFrame with columns ``epoch`` and ``best_vl_acc``
    (percent), sorted by epoch. Returns an **empty** DataFrame when the
    fold-1 log is absent.
    """
    df = per_run_epoch_log(f"kfold_{run_name}_fold1of5")
    if df.empty:
        return pd.DataFrame(columns=["epoch", "best_vl_acc"])
    recs = df.to_dict("records")
    return pd.DataFrame(
        {
            "epoch": [int(r["epoch"]) for r in recs],
            "best_vl_acc": [float(r["best_vl_acc"]) * 100.0 for r in recs],
        }
    ).sort_values("epoch").reset_index(drop=True)


def _panda_label(run_name: str, short_name: str) -> str:
    """Build the legend label ``"<short_name> (<5-fold mean>%)``" for a run.

    The value is the five-fold mean accuracy from the ``kfold_<run_name>.json``
    summary (CSV-derived, never hard-coded), so the label can never desync
    from the data.
    """
    kf = kfold_summary(run_name)
    if kf is None:
        raise AssertionError(f"no kfold summary for {run_name}")
    mean = float(kf["mean_val_acc"]) * 100.0
    return f"{short_name} ({mean:.2f}%)"


def _canonical_fold1():
    """Return ``(traj, (point, lo, hi))`` for the canonical run.

    ``traj`` is the fold-1 epoch trajectory (``epoch`` / ``best_vl_acc``
    percent); the tuple is the across-folds mean and 95% fold-bootstrap CI
    (percent) from :func:`paper.figures.loaders.kfold_acc_ci`, rounded to
    2dp for plotting.
    """
    traj = _fold1_trajectory(_CANON_RUN)
    ci = kfold_acc_ci(_CANON_RUN)
    if ci is None:
        raise AssertionError(f"no kfold acc CI for {_CANON_RUN}")
    point, lo, hi = (round(v, 2) for v in ci)
    return traj, (point, lo, hi)


def _build_fig():
    """Build the Figure 4 ``Figure`` and return ``(fig, (ax_a, ax_b))``.

    Split out from :func:`build` so the layout can be inspected
    programmatically (see :func:`verify`).
    """
    panda_color = style.DATASET_COLORS["PANDA"]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7, 6.0), dpi=300
    )

    # --- Panel (a): fold-1 validation accuracy vs epoch -----------------
    # PANDA fold-1 curves (per-epoch best_vl_acc); legend labels carry the
    # five-fold means from the kfold summaries.
    for run_name, short_name, ls in _PANDA_CURVES:
        traj = _fold1_trajectory(run_name)
        if traj.empty:
            continue
        ax_a.plot(traj["epoch"], traj["best_vl_acc"], color=panda_color,
                  linestyle=ls, linewidth=1.4, zorder=3,
                  label=_panda_label(run_name, short_name))

    # Canonical fold-1 trajectory + horizontal 5-fold-mean line + CI band.
    c_traj, (c_point, c_lo, c_hi) = _canonical_fold1()
    ax_a.fill_between([0.5, 33.5], c_lo, c_hi, color="black", alpha=0.10,
                       zorder=1, label=None)
    ax_a.axhline(c_point, color="black", linestyle="--", linewidth=1.0,
                 zorder=2)
    ax_a.plot(c_traj["epoch"], c_traj["best_vl_acc"], color="black",
              linestyle=_CANON_LINESTYLE, linewidth=1.4, zorder=3,
              label=f"canonical fold-1 probe ({c_point:.2f}% 5-fold mean)")
    # Label just above the mean line (transData: y is a data value).
    ax_a.text(0.5, c_point + 0.8, f"5-fold mean {c_point:.2f}%",
              transform=ax_a.transData, ha="left", va="bottom",
              fontsize=7, color="black")

    ax_a.set_xlabel("Epoch", fontweight="bold")
    ax_a.set_ylabel("Validation Accuracy (%)", fontweight="bold")
    ax_a.set_title("(a) PANDA val. accuracy (fold 1)",
                   fontweight="bold", loc="left")
    ax_a.set_xlim(0.5, 33.5)
    ax_a.set_ylim(0, 100)
    ax_a.grid(True, linestyle="--", alpha=style.GRID_ALPHA)

    # One frameless legend for the four curves, anchored upper-right inside
    # the axes but pulled down so it sits in the free band above the curves.
    ax_a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.85),
                fontsize=7, frameon=False)

    # --- Panel (b): GradNorm task-weight dynamics -----------------------
    # Seeded canonical probe log (seed 42); the title discloses provenance.
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
    ax_b.set_title("(b) GradNorm task weights (seed 42)",
                   fontweight="bold", loc="right")
    ax_b.set_xlim(0.5, 15.5)
    ax_b.set_ylim(0, 2.0)
    ax_b.grid(True, linestyle="--", alpha=style.GRID_ALPHA)

    # Explicit margins (tight_layout fails here: the "5-fold mean" label
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
      1. All three PANDA fold-1 trajectories and the canonical fold-1
         trajectory (11 epochs) are present.
      2. Panel (a) legend labels equal the kfold five-fold means (2dp).
      3. Panel (a) horizontal reference line sits at the 5-fold mean
         (22.70, 2dp).
      4. Panel (a) shaded band spans the 95% fold-bootstrap CI
         [14.66, 30.73] (2dp).
      5. Panel (b) task-weight curves have final values matching the
         canonical probe log (weights sum to 2.0).
      6. Rendered PNG geometry is sane (width/height in [800, 6000] px,
         aspect h/w in [0.4, 3.0]).
    """
    # All three PANDA runs must have fold-1 trajectories.
    for run_name, _, _ in _PANDA_CURVES:
        traj = _fold1_trajectory(run_name)
        assert not traj.empty, f"{run_name} fold-1 epoch log is empty"

    c_traj, (c_point, c_lo, c_hi) = _canonical_fold1()
    assert not c_traj.empty, "canonical fold-1 epoch log is empty"
    assert int(c_traj["epoch"].iloc[-1]) == 11, (
        f"canonical fold-1 trajectory has {len(c_traj)} epochs, expected 11"
    )

    # Build the figure to ensure it renders without error.
    fig, (ax_a, ax_b) = _build_fig()
    fig.canvas.draw()

    # Panel (a): legend labels must equal the kfold five-fold means (2dp).
    legend = ax_a.get_legend()
    assert legend is not None, "panel (a) has no legend"
    labels = [t.get_text() for t in legend.get_texts()]
    for run_name, short_name, _ in _PANDA_CURVES:
        kf = kfold_summary(run_name)
        mean = float(kf["mean_val_acc"]) * 100.0
        expected = f"{short_name} ({mean:.2f}%)"
        assert expected in labels, (
            f"legend missing {expected!r}; got {labels}"
        )

    # Panel (a): horizontal reference line at the 5-fold mean (2dp).
    hline = None
    for ln in ax_a.get_lines():
        y = ln.get_ydata()
        if len(y) == 2 and y[0] == y[1] and abs(y[0] - c_point) < 1e-9:
            hline = ln
    assert hline is not None, f"no horizontal line at 5-fold mean {c_point}"

    # Panel (a): shaded band spanning [lo, hi] (2dp).
    band = None
    for pc in ax_a.collections:
        ys = pc.get_paths()[0].vertices[:, 1]
        if abs(ys.min() - c_lo) < 1e-9 and abs(ys.max() - c_hi) < 1e-9:
            band = pc
    assert band is not None, f"no shaded band [{c_lo}, {c_hi}]"

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

    can = canonical_gradnorm()
    final_seg = float(can["seg_weight"].iloc[-1])
    final_cls = float(can["cls_weight"].iloc[-1])
    assert abs(final_seg + final_cls - 2.0) <= 0.05, (
        f"task weights sum {final_seg + final_cls} != 2.0"
    )

    print(f"fig4 panel (a) 5-fold means: "
          f"03={float(kfold_summary('g1_panda_vgg16')['mean_val_acc']) * 100:.2f} "
          f"10={float(kfold_summary('g2_panda_mobilenet_v2')['mean_val_acc']) * 100:.2f} "
          f"18={float(kfold_summary('g4_panda_isolate_gn')['mean_val_acc']) * 100:.2f} "
          f"canonical={c_point:.2f} CI=[{c_lo:.2f}, {c_hi:.2f}]")
    print(f"fig4 panel (b) final weights: seg={final_seg:.3f} cls={final_cls:.3f}")

    # --- Geometry guard --------------------------------------------------
    # Save the figure to a temp PNG and assert the rendered pixel dimensions
    # are sane (the F4b defect: a mis-transformed text label blew the
    # tight-bbox up to a ~1:42 vertical sliver).
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
        "n_panda_curves": 3,
        "n_canonical_curves": 1,
        "n_seg_lines": n_seg,
        "n_cls_lines": n_cls,
        "legend_labels": labels,
        "canonical_5fold_mean": c_point,
        "canonical_ci_lo": c_lo,
        "canonical_ci_hi": c_hi,
        "final_seg_weight": final_seg,
        "final_cls_weight": final_cls,
        "png_width_px": w,
        "png_height_px": h,
        "png_aspect_h_over_w": aspect,
    }


if __name__ == "__main__":
    for p in build():
        print(f"Saved: {p}")
