"""paper.figures — publication figure infrastructure for the CBM manuscript.

This package is the single source of truth for figure styling and for
reading the *ground-truth* numeric values that figures must plot. It is
deliberately decoupled from any one figure: every figure module imports
:mod:`paper.figures.style` for the locked visual contract and
:mod:`paper.figures.loaders` for the data, so no numeric value is ever
re-derived or hard-coded inside a figure.

Modules
-------
style
    Locked matplotlib rcParams + fixed color/marker contract.
loaders
    Read-only readers for the results CSVs, the epoch-log JSONL, and the
    canonical GradNorm probe log, with ground-truth assertions.

Design rules
------------
* ``DATASET_COLORS`` / ``ENCODER_MARKERS`` are fixed assignments. They are
  never re-derived per figure; a figure that needs a color looks it up here.
* No figure may call ``ax.twinx()`` (or ``twiny``). Dual-axis panels are
  forbidden by the style contract; use a shared axis or a separate panel.
"""

__all__ = ["style", "loaders"]
