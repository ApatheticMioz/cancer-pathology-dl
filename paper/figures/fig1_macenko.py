"""Figure 1 — Visual and Quantitative Impact of Macenko Stain Normalization.

Port of the original ``paper/generate_figures.py::generate_figure_1`` onto the
``paper.figures`` package. The four panels are unchanged in layout and
content:

* (a) raw PANDA biopsy tile crop (native chromatin)
* (b) Macenko-normalized crop (optical-density filtered)
* (c) absolute texture-discrepancy heatmap
* (d) Macenko ON/OFF top-1 accuracy bars

What changed versus the original:

* All styling comes from :mod:`paper.figures.style` (``apply()`` once at
  import; the heatmap uses the locked ``HEATMAP_CMAP`` = ``cividis``; the
  dataset colors come from the locked ``DATASET_COLORS``).
* The panel-(d) bar values are **read from the fold-campaign k-fold
  summaries** (:func:`paper.figures.loaders.kfold_summary`), not
  hard-coded and not from the stale single-split results matrix. The two
  datasets are PANDA (``g2_panda_mobilenet_v2`` ON vs
  ``g5_panda_nomacenko`` OFF) and PanNuke (``g3_pannuke_mobilenet_v2_final``
  ON vs ``g5_pannuke_nomacenko`` OFF); each bar carries a fold-SD whisker
  from the summary's ``std_val_acc``.
* The image panels keep their pinned local ``data/`` paths (the raw and
  Macenko tiles are local artifacts, not CSV-sourced).

Output: ``paper/fig1_macenko.pdf`` + ``paper/fig1_macenko.png`` (via
:func:`paper.figures.style.save_figure`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from . import style
from .loaders import REPO_ROOT, kfold_summary

# Apply the locked style contract once for this module.
style.apply()

# ---------------------------------------------------------------------------
# Pinned image paths (local data artifacts — NOT CSV-sourced).
# ---------------------------------------------------------------------------

#: Raw PANDA biopsy tile (native chromatin). Pinned local path.
_RAW_TILE = REPO_ROOT / "data" / "PANDA" / "train_images" / "train_images" / "3dab3238ef15a3c5b3d43e0b777073a5.png"
#: Macenko-normalized version of the same tile. Pinned local path.
_MAC_TILE = REPO_ROOT / "data" / "PANDA" / "preprocessed_macenko_fixed" / "images" / "3dab3238ef15a3c5b3d43e0b777073a5.png"

#: Central crop size (pixels) showing glandular structure.
_CROP_SIZE = 140

# ---------------------------------------------------------------------------
# CSV-sourced bar data (panel d). Read by Run Label, never hard-coded.
# ---------------------------------------------------------------------------

#: (kfold run name, dataset key) for the Macenko-ON and Macenko-OFF bars.
#: PANDA: ``g2_panda_mobilenet_v2`` (Mac ON) vs ``g5_panda_nomacenko``
#: (Mac OFF). PanNuke: ``g3_pannuke_mobilenet_v2_final`` (Mac ON) vs
#: ``g5_pannuke_nomacenko`` (Mac OFF). All four are the mobilenet_v2
#: v2-phase fold-campaign runs.
_MAC_ON = {
    "PANDA": "g2_panda_mobilenet_v2",
    "PANNUKE": "g3_pannuke_mobilenet_v2_final",
}
_MAC_OFF = {
    "PANDA": "g5_panda_nomacenko",
    "PANNUKE": "g5_pannuke_nomacenko",
}


def _kfold_bar(run_name: str) -> tuple[float, float]:
    """(mean_val_acc %, fold SD %) from the kfold summary (raises if absent)."""
    kf = kfold_summary(run_name)
    if kf is None:
        raise KeyError(f"no kfold summary for {run_name!r}")
    return (float(kf["mean_val_acc"]) * 100.0,
            float(kf["std_val_acc"]) * 100.0)


def _bars():
    """Return (labels, on_acc, on_sd, off_acc, off_sd) for the two datasets.

    ``on_sd`` / ``off_sd`` are the across-folds standard deviations
    (percentage points) from the kfold summaries — the fold-SD whiskers.
    """
    labels = []
    on_acc, on_sd, off_acc, off_sd = [], [], [], []
    for ds in ("PANDA", "PANNUKE"):
        labels.append(ds)
        on_acc.append(_kfold_bar(_MAC_ON[ds])[0])
        on_sd.append(_kfold_bar(_MAC_ON[ds])[1])
        off_acc.append(_kfold_bar(_MAC_OFF[ds])[0])
        off_sd.append(_kfold_bar(_MAC_OFF[ds])[1])
    return labels, on_acc, on_sd, off_acc, off_sd


def _build_fig():
    """Build the Figure 1 ``Figure`` and return ``(fig, axes)``.

    Split out from :func:`build` so the layout can be inspected
    programmatically (see :func:`verify`).
    """
    labels, on_acc, on_sd, off_acc, off_sd = _bars()

    # --- Image panels (a), (b), (c) -------------------------------------
    img_raw = np.array(Image.open(_RAW_TILE).convert("RGB"))
    img_mac = np.array(Image.open(_MAC_TILE).convert("RGB"))
    if img_raw.shape != img_mac.shape:
        img_mac = np.array(Image.fromarray(img_mac).resize((img_raw.shape[1], img_raw.shape[0])))

    h, w, _ = img_raw.shape
    ch, cw = h // 2, w // 2
    r1, r2 = ch - _CROP_SIZE // 2, ch + _CROP_SIZE // 2
    c1, c2 = cw - _CROP_SIZE // 2, cw + _CROP_SIZE // 2
    crop_raw = img_raw[r1:r2, c1:c2]
    crop_mac = img_mac[r1:r2, c1:c2]
    diff = np.mean(np.abs(crop_raw.astype(float) - crop_mac.astype(float)), axis=-1)

    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.7), dpi=300)

    axes[0].imshow(crop_raw)
    axes[0].set_title("(a) Raw Biopsy Tile\n(Native Chromatin)", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(crop_mac)
    axes[1].set_title("(b) Macenko-Normalized\n(Optical Density Filtered)", fontweight="bold")
    axes[1].axis("off")

    im_diff = axes[2].imshow(diff, cmap=style.HEATMAP_CMAP)
    axes[2].set_title("(c) Absolute Texture\nDiscrepancy Map", fontweight="bold")
    axes[2].axis("off")
    cbar = fig.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    # --- Panel (d): Macenko ON/OFF accuracy bars (CSV-sourced) ----------
    x = np.arange(len(labels))
    width = 0.32
    # Locked dataset colors (never re-derived per figure).
    on_color = [style.DATASET_COLORS[ds] for ds in labels]
    off_color = [style.DATASET_COLORS[ds] for ds in labels]

    # Whiskers: symmetric error bars from the across-folds SD (fold-SD).
    on_err = [on_sd, on_sd]
    off_err = [off_sd, off_sd]

    # Short legend labels ("Macenko ON" / "Macenko OFF") with neutral condition
    # keying: Macenko ON is hatched ('//') and Macenko OFF is solid, so the
    # condition is visibly encoded across both PANDA (orange) and PanNuke (purple)
    # bars without legend color conflation.
    from matplotlib.patches import Patch
    axes[3].bar(x - width / 2, on_acc, width, yerr=on_err, capsize=3,
                color=on_color, edgecolor="black", hatch="//",
                linewidth=0.8, zorder=3)
    axes[3].bar(x + width / 2, off_acc, width, yerr=off_err, capsize=3,
                color=off_color, edgecolor="black",
                linewidth=0.8, zorder=3)

    legend_elements = [
        Patch(facecolor="#D0D0D0", edgecolor="black", hatch="//", label="Macenko ON"),
        Patch(facecolor="#707070", edgecolor="black", label="Macenko OFF"),
    ]

    axes[3].set_ylabel("Top-1 Accuracy (%)", fontweight="bold")
    axes[3].set_title("(d) Ablation Accuracy Gain\n(Macenko ON vs OFF)", fontweight="bold")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels([f"{ds}\n(ISUP 0–5)" if ds == "PANDA" else f"{ds}\n(19 tissues)" for ds in labels])
    axes[3].set_ylim([20, 105])
    # Legend in the upper-left: the PANDA bars (left group) only reach ~40%,
    # so the upper-left corner (y > ~45) is clear headroom.
    axes[3].legend(handles=legend_elements, loc="upper left")
    axes[3].grid(axis="y", linestyle="--", alpha=style.GRID_ALPHA)

    # Delta labels on top of the OFF (raw) bars.
    for i in range(len(labels)):
        delta = off_acc[i] - on_acc[i]
        axes[3].annotate(f"+{delta:.2f}%",
                         xy=(x[i] + width / 2, off_acc[i] + 1.5),
                         ha="center", va="bottom", fontsize=8, fontweight="bold",
                         color=style.NULL_GRAY)

    plt.tight_layout()
    return fig, axes


def build() -> list:
    """Render Figure 1 and save PDF + PNG. Returns the written paths."""
    fig, _ = _build_fig()
    out_dir = REPO_ROOT / "paper"
    return style.save_figure(fig, out_dir, "fig1_macenko")


def _bbox_overlap(a, b, tol=0.0) -> bool:
    """True if two ``(x0, y0, x1, y1)`` bboxes overlap (with a small tol)."""
    return not (a[2] - tol <= b[0] or b[2] - tol <= a[0]
                or a[3] - tol <= b[1] or b[3] - tol <= a[1])


def verify() -> dict:
    """Programmatically verify the Figure 1 panel-(d) layout.

    Draws the canvas and checks that the panel-(d) legend does not overlap
    any bar (the previous "lower right" placement sat inside the tall
    PanNuke bars). Reports the legend bbox in data coords and the bar
    extents; raises ``AssertionError`` on any overlap.
    """
    fig, axes = _build_fig()
    ax = axes[3]
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    legend = ax.get_legend()
    leg_bbox = legend.get_window_extent(renderer)
    leg_data = ax.transData.inverted().transform(leg_bbox)
    leg_data = (float(leg_data[0][0]), float(leg_data[0][1]),
                float(leg_data[1][0]), float(leg_data[1][1]))

    # Bar containers only (skip ErrorbarContainer from the CI whiskers, which
    # has no ``.patches``). Each bar's data bbox.
    from matplotlib.container import BarContainer

    bar_bboxes = []
    for cont in ax.containers:
        if not isinstance(cont, BarContainer):
            continue
        for patch in cont.patches:
            x0, y0 = patch.get_x(), patch.get_y()
            x1, y1 = x0 + patch.get_width(), y0 + patch.get_height()
            bar_bboxes.append((float(x0), float(y0), float(x1), float(y1)))

    overlaps = [bb for bb in bar_bboxes if _bbox_overlap(leg_data, bb)]
    result = {
        "fig1d_legend_bbox_data": leg_data,
        "fig1d_legend_n_entries": len(legend.get_texts()),
        "fig1d_n_bars": len(bar_bboxes),
        "fig1d_legend_bar_overlaps": overlaps,
    }
    if overlaps:
        raise AssertionError(
            f"fig1d legend overlaps {len(overlaps)} bar(s): {overlaps}"
        )
    return result


if __name__ == "__main__":
    for p in build():
        print(f"Saved: {p}")
