"""Round-2 bootstrap Dice-CI whiskers for the Dice figure panels.

Self-contained (does NOT depend on the round-2 helpers in
:mod:`paper.figures.loaders`): reads ``results/kfold_campaign/dice_ci_summary.csv``
(written by ``scripts/bootstrap_dice_ci.py``) and exposes the 95%
percentile-bootstrap Dice-CI whisker for a run's ``overall`` stratum, in
*percent* (the unit the Dice panels plot).

The percentile-bootstrap math is re-implemented here and unit-checked
(:func:`verify_ci_math`) against the exact ground-truth algorithm in
``scripts/bootstrap_dice_ci.py`` so a figure never plots a CI it cannot
reproduce from the raw per-slice Dice.

Contract notes
--------------
* Dice CIs are strictly **percentile-bootstrap** (seed 42, 10,000
  resamples) — never Wilson. Wilson stays Acc-only (see
  ``scripts/compute_wilson_ci.py``); the two must never blur.
* The CSV stores fractions in [0, 1]; the Dice panels plot percentages, so
  :func:`whiskers_for_label` / :func:`whiskers_for_run` scale by 100.
* A run with no ``overall`` row (or no CI table yet) yields ``(nan, nan)``
  — callers draw the point without a whisker rather than fabricating one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (repo-root derived; no cwd assumptions)
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR: Path = REPO_ROOT / "results" / "kfold_campaign"
DICE_CI_SUMMARY_CSV: Path = CAMPAIGN_DIR / "dice_ci_summary.csv"
BOOTSTRAP_SCRIPT: Path = REPO_ROOT / "scripts" / "bootstrap_dice_ci.py"

#: Column order of the bootstrap Dice-CI table (scripts/bootstrap_dice_ci.py).
CI_COLUMNS = [
    "run_label", "stratum", "n", "point_estimate",
    "ci_low", "ci_high", "ci_width",
    "batch_mean_estimate", "batch_mean_available",
    "fold", "fold_sd",
]

#: The locked bootstrap seed / resample count (matches bootstrap_dice_ci.py).
BOOTSTRAP_SEED = 42
BOOTSTRAP_N_RESAMPLES = 10_000


# ---------------------------------------------------------------------------
# Ground-truth bootstrap algorithm (mirrors scripts/bootstrap_dice_ci.py)
# ---------------------------------------------------------------------------

def _load_bootstrap_module():
    """Load ``scripts/bootstrap_dice_ci.py`` as a module (path-based).

    The script is import-safe (its ``main`` is guarded by
    ``if __name__ == "__main__"``), so this gives us the exact ground-truth
    ``bootstrap_ci`` / ``load_records`` implementations to unit-check
    against.
    """
    spec = importlib.util.spec_from_file_location("bootstrap_dice_ci", BOOTSTRAP_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bootstrap_ci(values, n_resamples: int = BOOTSTRAP_N_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> tuple[float, float]:
    """95% percentile-bootstrap CI of the mean (deterministic for a fixed seed).

    Mirrors ``scripts/bootstrap_dice_ci.py::bootstrap_ci`` (numpy
    ``default_rng`` path). Returns ``(lo, hi)`` as fractions in [0, 1].
    """
    n = len(values)
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = np.asarray(values, dtype=float)[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


# ---------------------------------------------------------------------------
# Table loading + whisker lookup
# ---------------------------------------------------------------------------

def load_ci_table(campaign_dir: Path | None = None) -> pd.DataFrame:
    """Load ``<campaign>/dice_ci_summary.csv`` with numeric CI columns.

    Returns an empty DataFrame (with :data:`CI_COLUMNS`) when the file is
    absent. ``campaign_dir`` overrides the ``results/kfold_campaign`` root (e.g. a
    /tmp fixture dir) for standalone verification.
    """
    base = campaign_dir or CAMPAIGN_DIR
    path = base / "dice_ci_summary.csv"
    if not path.is_file():
        return pd.DataFrame(columns=CI_COLUMNS)
    df = pd.read_csv(path)
    for col in ("point_estimate", "ci_low", "ci_high", "ci_width"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _candidate_labels(run_num: int) -> list[str]:
    """Candidate round-2 run labels for a 1-based matrix run number.

    The per-run directory / CSV ``run_label`` may be the padded label
    (``03_g1_panda_vgg16``, the ``run_all_experiments.sh`` convention), the
    bare matrix run name (``g1_panda_vgg16``), or the kfold-campaign label
    (``kfold_g1_panda_vgg16``, the ``run_folds.sh`` convention). We try all
    three so the whisker lookup is robust to whichever campaign produced the
    dump.
    """
    from src.aggregate_results import EXPECTED_RUNS
    if run_num < 1 or run_num > len(EXPECTED_RUNS):
        raise IndexError(f"run_num {run_num} outside 1..{len(EXPECTED_RUNS)}")
    name = EXPECTED_RUNS[run_num - 1][1]
    return [f"{run_num:02d}_{name}", name, f"kfold_{name}"]


def whiskers_for_label(run_label: str, campaign_dir: Path | None = None) -> tuple[float, float]:
    """95% bootstrap Dice-CI whisker ``(lo, hi)`` in *percent* for a run.

    Sourced from the ``overall`` stratum row of the CI table. Returns
    ``(nan, nan)`` when the run has no overall row.
    """
    df = load_ci_table(campaign_dir)
    if df.empty:
        return (float("nan"), float("nan"))
    sel = df[(df["run_label"] == run_label) & (df["stratum"] == "overall")]
    if sel.empty:
        return (float("nan"), float("nan"))
    r = sel.iloc[0]
    return (float(r["ci_low"]) * 100.0, float(r["ci_high"]) * 100.0)


def whiskers_for_run(run_num: int, campaign_dir: Path | None = None) -> tuple[float, float]:
    """95% bootstrap Dice-CI whisker (percent) for a 1-based matrix run number.

    Tries the padded / bare / kfold label forms (see :func:`_candidate_labels`).
    Returns ``(nan, nan)`` when none of them has an overall row.
    """
    df = load_ci_table(campaign_dir)
    if df.empty:
        return (float("nan"), float("nan"))
    for label in _candidate_labels(run_num):
        sel = df[(df["run_label"] == label) & (df["stratum"] == "overall")]
        if not sel.empty:
            r = sel.iloc[0]
            return (float(r["ci_low"]) * 100.0, float(r["ci_high"]) * 100.0)
    return (float("nan"), float("nan"))


# ---------------------------------------------------------------------------
# Unit check of the CI-whisker math
# ---------------------------------------------------------------------------

def verify_ci_math(campaign_dir: Path | None = None, tol: float = 1e-5) -> dict:
    """Unit-check the CI-whisker math against the ground-truth algorithm.

    For every run dir under ``campaign_dir`` that has a
    ``per_slice_dice.jsonl``, recompute the 95% percentile-bootstrap CI of
    the per-case Dice mean using the *exact* ground-truth algorithm from
    ``scripts/bootstrap_dice_ci.py`` and assert it matches the ``ci_low`` /
    ``ci_high`` the ``dice_ci_summary.csv`` records (the values the figures
    plot). The CSV is written with 6-decimal rounding, so ``tol`` defaults to
    1e-5.

    Returns a per-run dict ``{run_label: {n, recomputed_lo, recomputed_hi,
    csv_lo, csv_hi, match}}``. Raises ``AssertionError`` if any run's
    recomputed CI disagrees with the CSV beyond ``tol``.
    """
    base = campaign_dir or CAMPAIGN_DIR
    bmod = _load_bootstrap_module()
    table = load_ci_table(base)
    out: dict[str, dict] = {}
    if not base.is_dir():
        return out

    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        dump = d / "per_slice_dice.jsonl"
        if not dump.is_file():
            continue
        recs = bmod.load_records(dump)
        vals = [r["dice"] for r in recs]
        if not vals:
            continue
        lo, hi = bmod.bootstrap_ci(vals, n_resamples=BOOTSTRAP_N_RESAMPLES,
                                   seed=BOOTSTRAP_SEED)
        csv_lo = csv_hi = None
        if not table.empty:
            sel = table[(table["run_label"] == d.name) & (table["stratum"] == "overall")]
            if not sel.empty:
                csv_lo = float(sel.iloc[0]["ci_low"])
                csv_hi = float(sel.iloc[0]["ci_high"])
        match = (
            csv_lo is not None
            and abs(lo - csv_lo) <= tol
            and abs(hi - csv_hi) <= tol
        )
        out[d.name] = {
            "n": len(vals),
            "recomputed_lo": lo,
            "recomputed_hi": hi,
            "csv_lo": csv_lo,
            "csv_hi": csv_hi,
            "match": match,
        }
        if not match:
            raise AssertionError(
                f"CI-whisker math mismatch for {d.name}: "
                f"recomputed [{lo:.6f}, {hi:.6f}] vs csv "
                f"[{csv_lo}, {csv_hi}] (tol={tol})"
            )
    return out
