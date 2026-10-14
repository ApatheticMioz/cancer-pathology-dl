"""Build all paper figures.

CLI::

    python -m paper.figures.build_all

Runs from the repository root (no cwd assumptions — every path is resolved
via :data:`paper.figures.loaders.REPO_ROOT`). Applies the locked style
contract once, then renders each figure module and saves its PDF + PNG pair
into ``paper/``.

Distilled-repo behavior
-----------------------
Two figures are *auxiliary*: they require raw training artifacts that a
distilled distribution of this repo does not ship — ``fig4`` plots the
per-fold ``epoch_log.jsonl`` trajectory logs and ``fig8`` runs forward-pass
inference from raw ``best.pt`` model checkpoints (which also pulls in the
training stack). In a distilled environment those raw artifacts are absent,
so both figures are **skipped with an informative notice** (instead of
failing the build). The pipeline exits ``0`` as long as every
publication-embedded figure (``fig1``, ``fig2``, ``fig3``, ``ga``, and the
non-auxiliary ``fig5``/``fig6``) succeeds.

Skips are detected *deterministically* up front (a missing-log /
missing-checkpoint precondition) rather than by guessing exception types; a
narrow :class:`FileNotFoundError` / :class:`ModuleNotFoundError` catch for
auxiliary figures is kept as defense-in-depth.

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


# ---------------------------------------------------------------------------
# Figure classification (publication-embedded vs auxiliary)
# ---------------------------------------------------------------------------

#: Figures that require **raw training artifacts** a distilled repo omits:
#:   * fig4 — per-fold ``epoch_log.jsonl`` trajectory logs
#:   * fig8 — raw ``best.pt`` model checkpoints (+ the training import stack)
#: When those artifacts are absent the figure is skipped with a notice.
_AUXILIARY_FIGURES = frozenset({"fig4", "fig8"})

#: Figures embedded in the publication; all of them (plus non-auxiliary
#: fig5/fig6) must succeed for the pipeline to exit 0.
_PUBLICATION_FIGURES = frozenset({"fig1", "fig2", "fig3", "ga"})


def _builder_map() -> dict:
    """Name -> ``(module, title)`` for every figure the build knows about."""
    return {
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


# ---------------------------------------------------------------------------
# Distilled-repo precondition checks (missing raw artifacts)
# ---------------------------------------------------------------------------

def _fig4_has_raw_logs() -> bool:
    """True when the per-fold ``epoch_log.jsonl`` trajectories fig4 plots exist.

    fig4 panel (a) draws the fold-1 trajectory for each PANDA curve plus the
    canonical run (``kfold_<run>_fold1of5/epoch_log.jsonl``). All of them must
    be present for a valid figure.
    """
    from . import loaders

    runs = [r for r, _s, _l in fig4_panda_anatomy._PANDA_CURVES]
    runs.append(fig4_panda_anatomy._CANON_RUN)
    for run in runs:
        if loaders.per_run_epoch_log(f"kfold_{run}_fold1of5").empty:
            return False
    return True


def _fig8_has_checkpoints() -> bool:
    """True when every fig8 run's raw ``best.pt`` checkpoint is present.

    Reuses :func:`fig8_qualitative._resolve_checkpoint` so the path
    resolution (incl. the stale ``round2`` -> ``kfold_campaign`` migration)
    stays a single source of truth.
    """
    for run in fig8_qualitative._RUNS:
        try:
            ckpt = fig8_qualitative._resolve_checkpoint(run)
        except (FileNotFoundError, KeyError):
            return False
        if not ckpt.exists():
            return False
    return True


def _skip_reason(name: str) -> str | None:
    """Return a skip notice for a distilled-repo auxiliary figure, else None.

    A non-auxiliary figure never skips. An auxiliary figure skips only when
    its raw training artifacts are absent from this (distilled) repo.
    """
    if name not in _AUXILIARY_FIGURES:
        return None
    if name == "fig4" and not _fig4_has_raw_logs():
        return ("Skipping auxiliary figure: per-epoch jsonl training logs not "
                "present in distilled repo")
    if name == "fig8" and not _fig8_has_checkpoints():
        return ("Skipping auxiliary figure: raw training checkpoints (best.pt) "
                "not present in distilled repo")
    return None


def _is_missing_raw_artifact(exc: BaseException) -> bool:
    """True when an exception reflects absent raw training artifacts.

    :class:`FileNotFoundError` (a missing log/checkpoint) or a
    :class:`ModuleNotFoundError` (the heavy training import stack a distilled
    env omits) is the distilled-repo signal rather than a code defect.
    """
    return isinstance(exc, (FileNotFoundError, ModuleNotFoundError))


# ---------------------------------------------------------------------------
# Build / verify
# ---------------------------------------------------------------------------

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


def _build_one(name: str, mod, title: str, verify: bool) -> bool:
    """Build (and optionally verify) a single figure; return True on success.

    Auxiliary figures whose raw training artifacts are absent from this
    (distilled) repo are **skipped with an informative notice** and count as
    success — they cannot render without the raw data, and their absence is
    expected in a distilled distribution. A :class:`FileNotFoundError` /
    :class:`ModuleNotFoundError` raised while building such a figure is
    treated the same way (defense-in-depth). Any other failure returns False.
    """
    print(f"==> {title}")

    reason = _skip_reason(name)
    if reason is not None:
        print(f"    [Notice] {reason}")
        return True

    try:
        paths = mod.build()
    except Exception as exc:  # noqa: BLE001 - report (or skip) and continue
        if name in _AUXILIARY_FIGURES and _is_missing_raw_artifact(exc):
            print(f"    [Notice] Skipping auxiliary figure: raw training "
                  f"artifacts not present in distilled repo ({exc!r})")
            return True
        print(f"    ERROR building {name}: {exc!r}", file=sys.stderr)
        return False

    for p in paths:
        print(f"    Saved: {p}")

    if verify:
        return _verify(mod, name)
    return True


def _run_build(names: list | None, verify: bool) -> int:
    """Build the given figures (all when ``names`` is None); return exit code."""
    builders = _builder_map()
    wanted = sorted(names) if names is not None else list(builders)

    failures = 0
    for name in wanted:
        if name not in builders:
            print(f"Unknown figure: {name}", file=sys.stderr)
            failures += 1
            continue
        mod, title = builders[name]
        if not _build_one(name, mod, title, verify):
            failures += 1

    return 1 if failures else 0


def build_all(verify: bool = True) -> int:
    """Render every figure (and verify layout); return a process exit code."""
    style.apply()  # locked style contract, applied once for the whole build
    return _run_build(None, verify)


def main(argv: list) -> int:
    style.apply()
    do_verify = "--no-verify" not in argv
    args = [a for a in argv[1:] if a != "--no-verify"]

    if args:
        wanted = set(args)
        known = set(_builder_map())
        unknown = wanted - known
        if unknown:
            print(f"Unknown figure(s): {sorted(unknown)}; valid: {sorted(known)}",
                  file=sys.stderr)
            return 2
        return _run_build(sorted(wanted), do_verify)

    return build_all(verify=do_verify)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
