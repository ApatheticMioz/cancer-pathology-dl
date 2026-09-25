"""Build all paper figures.

CLI::

    python -m paper.figures.build_all

Runs from the repository root (no cwd assumptions — every path is resolved
via :data:`paper.figures.loaders.REPO_ROOT`). Applies the locked style
contract once, then renders each figure module and saves its PDF + PNG pair
into ``paper/``.

After building, each figure's layout is verified programmatically (no visual
read-back): the fig2 panel-(c) legend must not overlap any data line and the
y-limits must expose the 44.08 curve anchor; the fig1 panel-(d) legend must
not overlap any bar. The final axes limits and legend bbox for fig2c are
printed.

Usage::

    python -m paper.figures.build_all            # build + verify every figure
    python -m paper.figures.build_all fig1       # build only Figure 1
    python -m paper.figures.build_all fig2       # build only Figure 2
    python -m paper.figures.build_all fig3       # build only Figure 3
    python -m paper.figures.build_all fig4       # build only Figure 4
    python -m paper.figures.build_all fig5       # build only Figure 5
    python -m paper.figures.build_all fig6       # build only Figure 6
    python -m paper.figures.build_all fig8       # build only Figure 8
    python -m paper.figures.build_all ga         # build only the graphical abstract
    python -m paper.figures.build_all --no-verify  # build without layout checks
"""

from __future__ import annotations

import sys

from . import style
from . import fig1_macenko, fig2_empty_dice, fig3_results_dotplot
from . import fig4_panda_anatomy
from . import fig5_ablation_forest
from . import fig6_lambda_sweep
from . import fig8_qualitative
from . import graphical_abstract


def _verify(mod, name: str) -> bool:
    """Run a figure module's ``verify()`` and print the result.

    Returns True on success, False on failure (prints the reason).
    """
    if not hasattr(mod, "verify"):
        return True
    try:
        result = mod.verify()
    except AssertionError as exc:
        print(f"    VERIFY FAIL [{name}]: {exc}", file=sys.stderr)
        return False
    print(f"    VERIFY OK [{name}]:")
    for key, val in result.items():
        print(f"        {key}: {val}")
    return True


def build_all(verify: bool = True) -> int:
    """Render every figure (and verify layout); return a process exit code."""
    style.apply()  # locked style contract, applied once for the whole build

    builders = {
        "fig1": (fig1_macenko, "Figure 1: Macenko Stain Normalization"),
        "fig2": (fig2_empty_dice, "Figure 2: Empty-Mask Dice Inflation"),
        "fig3": (fig3_results_dotplot,
                 "Figure 3: 27-Config Claimed-vs-Measured CI Dot Plot"),
        "fig4": (fig4_panda_anatomy,
                 "Figure 4: PANDA Degradation Anatomy + GradNorm Weights"),
        "fig5": (fig5_ablation_forest,
                 "Figure 5: Matched-Ablation Forest Plot"),
        "fig6": (fig6_lambda_sweep,
                 "Figure 6: Lambda Loss-Weight Ratio Sweep"),
        "fig8": (fig8_qualitative,
                 "Figure 8: Qualitative GT-vs-Prediction Validation Tiles"),
        "ga": (graphical_abstract,
               "Graphical Abstract: Wide CBM Banner (Claimed/Why/Audit)"),
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
        if verify:
            if not _verify(mod, name):
                failures += 1
    return 1 if failures else 0


def main(argv: list) -> int:
    style.apply()
    do_verify = "--no-verify" not in argv
    args = [a for a in argv[1:] if a != "--no-verify"]

    if args:
        wanted = set(args)
        known = {"fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig8", "ga"}
        unknown = wanted - known
        if unknown:
            print(f"Unknown figure(s): {sorted(unknown)}; valid: {sorted(known)}",
                  file=sys.stderr)
            return 2
        builders = {
            "fig1": (fig1_macenko, "Figure 1: Macenko Stain Normalization"),
            "fig2": (fig2_empty_dice, "Figure 2: Empty-Mask Dice Inflation"),
            "fig3": (fig3_results_dotplot,
                     "Figure 3: 27-Config Claimed-vs-Measured CI Dot Plot"),
            "fig4": (fig4_panda_anatomy,
                     "Figure 4: PANDA Degradation Anatomy + GradNorm Weights"),
            "fig5": (fig5_ablation_forest,
                     "Figure 5: Matched-Ablation Forest Plot"),
            "fig6": (fig6_lambda_sweep,
                     "Figure 6: Lambda Loss-Weight Ratio Sweep"),
            "fig8": (fig8_qualitative,
                     "Figure 8: Qualitative GT-vs-Prediction Validation Tiles"),
            "ga": (graphical_abstract,
                   "Graphical Abstract: Wide CBM Banner (Claimed/Why/Audit)"),
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
                continue
            if do_verify and not _verify(mod, name):
                failures += 1
        return 1 if failures else 0

    return build_all(verify=do_verify)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
