"""Figure 5 — Matched-ablation forest plot (validation accuracy + 95% CIs).

One panel (~7in wide x ~4.5in tall, one axis, NO twinx). Six *matched*
ablation pairs are drawn as a forest plot: each pair has two arms (arm A on
top, arm B below) with a small gap between pairs. Each arm is a marker
(color = dataset, shape = encoder) with a horizontal 95% Wilson-CI whisker
and a 7pt annotation of the run number + value (e.g. ``R18 29.04``).

The six pairs (all PANDA/TCGA/PanNuke, from
:func:`paper.figures.loaders.results_matrix`):

* 18 vs 20 — isolated GradNorm (29.04) vs static 5:1 (45.39) — **disjoint**
* 17 vs 20 — LR 1e-4 (43.35) vs 1e-3 (45.39) — **overlap**
* 25 vs 08 — no-skip TCGA (94.34) vs matched (93.32) — **overlap**
* 26 vs 10 — no-skip PANDA (35.31) vs matched (34.70) — **overlap**
* 23 vs 10 — no-Mac PANDA (40.21) vs matched (34.70) — **disjoint**
* 24 vs 16 — no-Mac PanNuke (99.36) vs matched (96.68) — **disjoint**

Verdict encoding (NOT color-alone): a right-margin column carries a text
verdict per pair — ``CIs disjoint`` vs ``CIs overlap`` — recomputed from the
CI bounds (overlap iff ``a_lo <= b_hi and b_lo <= a_hi``). Disjoint pairs
also get a subtle :data:`style.NULL_GRAY` band behind their two rows;
overlapping pairs get no band.

Output: ``paper/fig5_ablation_forest.pdf`` + ``paper/fig5_ablation_forest.png``
(via :func:`paper.figures.style.save_figure`).
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import style
from .loaders import REPO_ROOT, results_matrix

# Apply the locked style contract once for this module.
style.apply()

# ---------------------------------------------------------------------------
# Matched-pair definitions.
# ---------------------------------------------------------------------------

#: (pair label, (run A, run B)) in display order (top -> bottom).
#: Arm A is the "treatment / ablated" arm (top); arm B is the matched control
#: (bottom). Run numbers are 1-based CSV row indices.
_PAIRS = [
    ("GradNorm on/off (PANDA)", (18, 20)),
    ("LR 1e-4 vs 1e-3 (PANDA)", (17, 20)),
    ("Skip connections (TCGA)", (25, 8)),
    ("Skip connections (PANDA)", (26, 10)),
    ("Macenko removal (PANDA)", (23, 10)),
    ("Macenko removal (PanNuke)", (24, 16)),
]

#: Vertical spacing between pair centers (in y-data units).
_PAIR_SPACING = 2.0
#: Half the vertical offset of each arm from its pair center.
_ARM_OFFSET = 0.5


def _arm_data(run_num: int):
    """Return ``(dataset, encoder, acc, ci_lo, ci_hi)`` for one run."""
    row = results_matrix().iloc[run_num - 1]
    return (
        str(row["Dataset"]),
        str(row["Encoder"]),
        float(row["Accuracy (%)"]),
        float(row["Acc 95% CI Lower"]),
        float(row["Acc 95% CI Upper"]),
    )


def _ci_overlap(a_lo, a_hi, b_lo, b_hi) -> bool:
    """True iff the two 95% CIs overlap (``a_lo <= b_hi and b_lo <= a_hi``)."""
    return a_lo <= b_hi and b_lo <= a_hi


def _build_fig():
    """Build the Figure 5 ``Figure`` and return ``(fig, ax)``.

    Split out from :func:`build` so the layout can be inspected
    programmatically (see :func:`verify`).
    """
    n_pairs = len(_PAIRS)
    # Pair centers: top pair at the top, descending.
    pair_centers = [(n_pairs - 1 - i) * _PAIR_SPACING for i in range(n_pairs)]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), dpi=300)

    # --- Disjoint-pair bands (drawn first, behind everything) -----------
    for (label, (ra, rb)), center in zip(_PAIRS, pair_centers):
        da, ea, va, alo, ahi = _arm_data(ra)
        db, eb, vb, blo, bhi = _arm_data(rb)
        if not _ci_overlap(alo, ahi, blo, bhi):
            # Subtle band behind the two rows of a disjoint pair.
            ax.axhspan(center - _ARM_OFFSET - 0.1, center + _ARM_OFFSET + 0.1,
                       color=style.NULL_GRAY, alpha=0.10, zorder=0)

    # --- Arms: marker + CI whisker + annotation ------------------------
    for (label, (ra, rb)), center in zip(_PAIRS, pair_centers):
        for run_num, y in ((ra, center + _ARM_OFFSET),
                           (rb, center - _ARM_OFFSET)):
            ds, enc, val, lo, hi = _arm_data(run_num)
            color = style.DATASET_COLORS[ds]
            marker = style.ENCODER_MARKERS[enc]
            # Horizontal CI whisker with end caps.
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.2,
                    zorder=2, solid_capstyle="butt")
            cap = 0.12
            ax.plot([lo, lo], [y - cap, y + cap], color=color, linewidth=1.2,
                    zorder=2)
            ax.plot([hi, hi], [y - cap, y + cap], color=color, linewidth=1.2,
                    zorder=2)
            # Marker at the measured value.
            ax.plot(val, y, marker=marker, color=color, markersize=6,
                    linestyle="none", zorder=3)
            # 7pt annotation of run number + value, next to the marker.
            # Left-side arms (PANDA, < 80) annotate to the right; right-side
            # arms (TCGA/PanNuke, >= 80) annotate to the left to avoid the
            # right edge.
            if val < 80:
                ax.annotate(f"R{run_num:02d} {val:.2f}",
                            xy=(val, y), xytext=(8, 0),
                            textcoords="offset points", fontsize=7,
                            ha="left", va="center", color="black")
            else:
                ax.annotate(f"R{run_num:02d} {val:.2f}",
                            xy=(val, y), xytext=(-8, 0),
                            textcoords="offset points", fontsize=7,
                            ha="right", va="center", color="black")

    # --- Axes -----------------------------------------------------------
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-_ARM_OFFSET - 0.3,
                (n_pairs - 1) * _PAIR_SPACING + _ARM_OFFSET + 0.3)
    ax.set_xlabel("Validation Accuracy (%)", fontweight="bold")
    ax.set_title("Matched Ablation Pairs (95% Wilson CIs)", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=style.GRID_ALPHA)

    # --- Left-margin pair labels + right-margin verdicts ---------------
    # Explicit margins leave room for both the left pair labels and the
    # right verdict column.
    fig.subplots_adjust(left=0.30, right=0.80, top=0.90, bottom=0.12)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pos = ax.get_position()
    ylim = ax.get_ylim()

    def _y_frac(y_data):
        return pos.y0 + (y_data - ylim[0]) / (ylim[1] - ylim[0]) * pos.height

    for (label, (ra, rb)), center in zip(_PAIRS, pair_centers):
        da, ea, va, alo, ahi = _arm_data(ra)
        db, eb, vb, blo, bhi = _arm_data(rb)
        overlap = _ci_overlap(alo, ahi, blo, bhi)
        verdict = "CIs overlap" if overlap else "CIs disjoint"
        # Left-margin pair label.
        fig.text(pos.x0 - 0.015, _y_frac(center), label, ha="right",
                 va="center", fontsize=7, color="black")
        # Right-margin verdict (text, not color-alone).
        fig.text(pos.x1 + 0.015, _y_frac(center), verdict, ha="left",
                 va="center", fontsize=7,
                 color="black" if not overlap else style.NULL_GRAY)

    return fig, ax


def build() -> list:
    """Render Figure 5 and save PDF + PNG. Returns the written paths."""
    fig, _ = _build_fig()
    out_dir = REPO_ROOT / "paper"
    return style.save_figure(fig, out_dir, "fig5_ablation_forest")


def verify() -> dict:
    """Verify the Figure 5 data contract (no visual read-back).

    Asserts:
      * 6 pairs / 12 arms are present;
      * every arm has parsed 95% CI bounds (not NaN);
      * each pair's rendered verdict text matches the overlap recomputed
        from the CI bounds (overlap iff ``a_lo <= b_hi and b_lo <= a_hi``).
    Prints the verdicts table. Returns a summary dict.
    """
    fig, ax = _build_fig()
    fig.canvas.draw()

    # Count the arms: each arm has exactly one "R<nn> <value>" annotation.
    n_arms = sum(
        1 for t in ax.texts
        if t.get_text().startswith("R") and len(t.get_text().split()) == 2
    )
    n_pairs = len(_PAIRS)
    assert n_pairs == 6, f"expected 6 pairs, got {n_pairs}"
    assert n_arms == 12, f"expected 12 arm annotations, got {n_arms}"

    # Every arm has CI bounds; recompute each pair's verdict from the CIs.
    verdicts = {}
    n_arms_checked = 0
    for label, (ra, rb) in _PAIRS:
        da, ea, va, alo, ahi = _arm_data(ra)
        db, eb, vb, blo, bhi = _arm_data(rb)
        for run_num, (lo, hi) in ((ra, (alo, ahi)), (rb, (blo, bhi))):
            assert not (np.isnan(lo) or np.isnan(hi)), (
                f"arm R{run_num:02d} has NaN CI bounds"
            )
            n_arms_checked += 1
        overlap = _ci_overlap(alo, ahi, blo, bhi)
        verdicts[label] = "CIs overlap" if overlap else "CIs disjoint"
    assert n_arms_checked == 12, f"expected 12 arms with CIs, got {n_arms_checked}"

    # The rendered right-margin verdict texts must match the recomputed
    # overlap (read them back from the figure's text artists).
    rendered = {t.get_text() for t in fig.texts}
    for label, verdict in verdicts.items():
        assert verdict in rendered, (
            f"verdict {verdict!r} for {label!r} not found among rendered texts"
        )

    print("fig5 verdicts:")
    for label, verdict in verdicts.items():
        print(f"    {label:28s} {verdict}")

    plt.close(fig)

    return {
        "n_pairs": n_pairs,
        "n_arms": n_arms,
        "n_arms_with_ci": n_arms_checked,
        "verdicts": verdicts,
    }


if __name__ == "__main__":
    for p in build():
        print(f"Saved: {p}")
