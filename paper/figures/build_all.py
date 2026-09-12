"""Build all paper figures.

CLI::

    python -m paper.figures.build_all

Runs from the repository root (no cwd assumptions — every path is resolved
via :data:`paper.figures.loaders.REPO_ROOT`). Applies the locked style
contract once, then renders each figure module and saves its PDF + PNG pair
into ``paper/``.

Usage::

    python -m paper.figures.build_all            # build every figure
    python -m paper.figures.build_all fig1       # build only Figure 1
    python -m paper.figures.build_all fig2       # build only Figure 2
"""

from __future__ import annotations

import sys

from . import style
from . import fig1_macenko, fig2_empty_dice


def build_all() -> int:
    """Render every figure; return a process exit code (0 on success)."""
    style.apply()  # locked style contract, applied once for the whole build

    builders = {
        "fig1": (fig1_macenko, "Figure 1: Macenko Stain Normalization"),
        "fig2": (fig2_empty_dice, "Figure 2: Empty-Mask Dice Inflation"),
    }

    failures = 0
    for name, (mod, title) in builders.items():
        print(f"==> {title}")
        try:
            paths = mod.build()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"    ERROR building {name}: {exc!r}", file=sys.stderr)
            continue
        for p in paths:
            print(f"    Saved: {p}")
    return 1 if failures else 0


def main(argv: list) -> int:
    style.apply()
    if len(argv) > 1:
        # Build only the requested subset.
        wanted = set(argv[1:])
        known = {"fig1", "fig2"}
        unknown = wanted - known
        if unknown:
            print(f"Unknown figure(s): {sorted(unknown)}; valid: {sorted(known)}",
                  file=sys.stderr)
            return 2
        builders = {
            "fig1": (fig1_macenko, "Figure 1: Macenko Stain Normalization"),
            "fig2": (fig2_empty_dice, "Figure 2: Empty-Mask Dice Inflation"),
        }
        failures = 0
        for name in sorted(wanted):
            mod, title = builders[name]
            print(f"==> {title}")
            try:
                for p in mod.build():
                    print(f"    Saved: {p}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"    ERROR building {name}: {exc!r}", file=sys.stderr)
        return 1 if failures else 0
    return build_all()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
