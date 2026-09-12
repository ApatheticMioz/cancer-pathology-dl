#!/usr/bin/env python3
"""DEPRECATED shim — figure generation now lives in the ``paper.figures`` package.

This module used to hard-code the two publication figures. It is now a thin
shim that delegates to :mod:`paper.figures.build_all`, which sources every
numeric value from the results CSVs / epoch logs and applies the locked
style contract from :mod:`paper.figures.style`.

Run the real builder directly::

    python -m paper.figures.build_all

or via the Makefile::

    make figures

Importing this module still works (for backward compatibility) and simply
re-exports the package builder.
"""

from __future__ import annotations

import sys

from paper.figures import build_all as _build_all
from paper.figures import style as _style

__all__ = ["generate_figure_1", "generate_figure_2", "main"]


def _deprecation_pointer() -> None:
    print(
        "NOTE: paper/generate_figures.py is a deprecated shim.\n"
        "      Figures are now built by the paper.figures package:\n"
        "          python -m paper.figures.build_all\n"
        "      (or: make figures)",
        file=sys.stderr,
    )


def generate_figure_1() -> None:
    """Deprecated: builds Figure 1 via the package."""
    _deprecation_pointer()
    _style.apply()
    from paper.figures import fig1_macenko

    for p in fig1_macenko.build():
        print(f"Saved: {p}")


def generate_figure_2() -> None:
    """Deprecated: builds Figure 2 via the package."""
    _deprecation_pointer()
    _style.apply()
    from paper.figures import fig2_empty_dice

    for p in fig2_empty_dice.build():
        print(f"Saved: {p}")


def main() -> int:
    """Build all figures via the package."""
    _deprecation_pointer()
    return _build_all.main(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
