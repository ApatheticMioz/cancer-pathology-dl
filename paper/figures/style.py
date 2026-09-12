"""Locked visual style contract for every paper figure.

This module is the *single source of truth* for figure appearance. A figure
module calls :func:`apply` once (typically at import time) and then draws
against the resulting ``matplotlib.rcParams``. Nothing in a figure may
override these values ad hoc; if a value needs to change, it changes here
and everywhere at once.

Fixed contract
--------------
``DATASET_COLORS``
    Fixed dataset -> color assignment. These are the Okabe-Ito
    color-blind-safe palette. The mapping is *locked*: it is never
    re-derived, re-sorted, or re-assigned per figure. A figure that needs a
    dataset's color looks it up here.
``ENCODER_MARKERS``
    Fixed encoder -> marker assignment (``vgg16`` -> circle,
    ``mobilenet_v2`` -> square).
``CLAIMED_STYLE``
    The style used for *reference / claimed* (published) values: black open
    diamond markers on a black dashed line.
``NULL_GRAY``
    Gray used for null / no-effect verdicts.
``HEATMAP_CMAP``
    The colormap used for all heatmaps (``cividis`` — perceptually uniform
    and color-blind safe).

``apply()``
    Sets the full ``matplotlib.rcParams`` block. Idempotent.

Style rules
-----------
* Font: DejaVu Sans.
* Base font size 8 pt; axis titles 9 pt, axis labels 8 pt, tick labels 7 pt,
  legend text 7 pt.
* Top and right spines are removed on every axis.
* Legends are frameless.
* Grid alpha 0.15.
* Line width 1.2.
* ``pdf.fonttype = 42`` (TrueType) so PDF text is selectable/editable.
* ``savefig`` writes PDF + PNG at 300 dpi with ``bbox_inches='tight'``.

NEVER use ``ax.twinx()`` (or ``twiny``) in any figure. Dual-axis panels are
forbidden by this contract; use a shared axis or a separate subplot instead.
"""

from __future__ import annotations

import matplotlib

# ---------------------------------------------------------------------------
# Fixed color / marker contract (locked — never re-derived per figure)
# ---------------------------------------------------------------------------

#: Dataset -> color. Okabe-Ito color-blind-safe palette. LOCKED assignment.
DATASET_COLORS = {
    "TCGA": "#0072B2",
    "PANDA": "#D55E00",
    "SIIM": "#009E73",
    "PANNUKE": "#CC79A7",
}

#: Encoder -> marker. LOCKED assignment.
ENCODER_MARKERS = {
    "vgg16": "o",
    "mobilenet_v2": "s",
}

#: Reference / claimed (published) values: black open diamond on black dashed.
CLAIMED_STYLE = {
    "marker": "D",
    "mfc": "none",
    "mec": "black",
    "color": "black",
    "linestyle": "--",
    "linewidth": 1.2,
}

#: Null / no-effect verdict color.
NULL_GRAY = "#6E6E6E"

#: Heatmap colormap (perceptually uniform, color-blind safe).
HEATMAP_CMAP = "cividis"

#: Grid alpha (locked).
GRID_ALPHA = 0.15

#: Save resolution for raster (PNG) output.
SAVE_DPI = 300


def apply() -> None:
    """Set the locked ``matplotlib.rcParams`` block. Idempotent.

    Call once per figure module (typically at import). Safe to call multiple
    times; it only ever writes the same values.
    """
    rc = matplotlib.rcParams

    # Font
    rc["font.family"] = "DejaVu Sans"
    rc["font.size"] = 8.0
    rc["axes.titlesize"] = 9.0
    rc["axes.labelsize"] = 8.0
    rc["xtick.labelsize"] = 7.0
    rc["ytick.labelsize"] = 7.0
    rc["legend.fontsize"] = 7.0

    # Spines: keep only left + bottom
    rc["axes.spines.top"] = False
    rc["axes.spines.right"] = False
    rc["axes.spines.left"] = True
    rc["axes.spines.bottom"] = True

    # Legend: frameless
    rc["legend.frameon"] = False

    # Grid
    rc["grid.alpha"] = 0.15
    rc["grid.linewidth"] = 0.6

    # Lines
    rc["lines.linewidth"] = 1.2

    # PDF: TrueType fonts (selectable / editable text)
    rc["pdf.fonttype"] = 42
    rc["ps.fonttype"] = 42

    # Savefig: PDF + PNG at 300 dpi, tight bbox
    rc["savefig.dpi"] = SAVE_DPI
    rc["savefig.bbox"] = "tight"
    rc["savefig.format"] = "pdf"

    # Figure defaults
    rc["figure.dpi"] = 100
    rc["figure.facecolor"] = "white"
    rc["axes.facecolor"] = "white"


def save_figure(fig, out_dir, base_name) -> list:
    """Save ``fig`` as both PDF and PNG under ``out_dir`` with ``base_name``.

    Returns the list of written file paths (PDF first, then PNG). Uses the
    locked 300 dpi / tight-bbox contract from :func:`apply`.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{base_name}.pdf"
    png_path = out_dir / f"{base_name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=SAVE_DPI)
    return [pdf_path, png_path]
