#!/usr/bin/env python3
"""Round-2 Dice confidence intervals from per-case dumps.

Reads ``results/round2/<run_label>/per_slice_dice.jsonl`` records with the
schema ``{run_label, dataset, encoder, fold, seed, case_id, dice,
empty_pred, empty_gt, label_int}`` and computes, per run:

  * point estimate = mean of the per-case ``dice`` values, recomputed from
    the records (NOT the rounded summary value);
  * mean-of-batch-means, if a summary JSON sits alongside the dump in the
    same directory (``batch_dice`` / ``dice_history`` / ``batches[].dice``
    keys are recognised).  Both values are reported side by side; they may
    differ when the last batch is short, and they are NEVER averaged
    together silently;
  * 95% confidence interval by percentile bootstrap: 10,000 resamples,
    seed 42, fully deterministic (numpy ``default_rng(42)`` when numpy is
    available, otherwise ``random.Random(42)`` with an identical
    linear-interpolation percentile);
  * three strata per run: ``overall``, ``positive_only`` (``empty_gt=False``)
    and ``negative_only`` (``empty_gt=True``), each with its own ``n``.

Output: ``<round2>/dice_ci_summary.csv`` (one row per run x stratum) plus a
stdout table.

``--fold-aware``: when a run has fold dump directories named
``<base>_fold<N>of5``, per-fold rows are emitted in addition to the base
rows, plus one ``across_folds`` row carrying the mean +/- SD of the
per-fold point estimates (CI bootstrapped over the fold means).

Dice CIs are strictly separate from the Wilson accuracy CIs produced by
``scripts/compute_wilson_ci.py`` (Wilson stays Acc-only) -- the two must
never blur.

Usage:
    python3 scripts/bootstrap_dice_ci.py [--fold-aware]
        [--round2-dir results/round2] [--out <csv>]
        [--n-resamples 10000] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROUND2 = PROJECT_ROOT / "results" / "round2"

FOLD_RE = re.compile(r"^(?P<base>.+)_fold(?P<n>\d+)of(?P<k>\d+)$")

CSV_COLUMNS = [
    "run_label",
    "stratum",
    "n",
    "point_estimate",
    "ci_low",
    "ci_high",
    "ci_width",
    "batch_mean_estimate",
    "batch_mean_available",
    "fold",
    "fold_sd",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    """Load per-case dice records from a JSONL dump."""
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(
                {
                    "dice": float(rec["dice"]),
                    "empty_gt": bool(rec["empty_gt"]),
                    "empty_pred": bool(rec.get("empty_pred", False)),
                    "case_id": rec.get("case_id"),
                    "fold": rec.get("fold"),
                }
            )
    return records


def batch_mean_from_summary(run_dir: Path) -> float | None:
    """Mean-of-batch-means from a summary JSON sitting alongside the dump.

    Recognised shapes: ``{"batch_dice": [..]}``, ``{"dice_history": [..]}``,
    ``{"batches": [{"dice": ..}, ..]}``, or a top-level list of
    ``{"dice": ..}`` dicts.  Returns None when no summary is found.
    """
    for path in sorted(run_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        batches: list[float] | None = None
        if isinstance(data, dict):
            for key in ("batch_dice", "dice_history", "dice_per_batch"):
                v = data.get(key)
                if (
                    isinstance(v, list)
                    and v
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
                ):
                    batches = [float(x) for x in v]
                    break
            if batches is None:
                b = data.get("batches")
                if (
                    isinstance(b, list)
                    and b
                    and all(isinstance(x, dict) and "dice" in x for x in b)
                ):
                    batches = [float(x["dice"]) for x in b]
        elif isinstance(data, list) and data and all(
            isinstance(x, dict) and "dice" in x for x in data
        ):
            batches = [float(x["dice"]) for x in data]
        if batches:
            return sum(batches) / len(batches)
    return None


def discover_runs(round2_dir: Path) -> dict[str, Path]:
    """Map run_label -> dump dir for every dir holding a per_slice_dice.jsonl."""
    runs: dict[str, Path] = {}
    for d in sorted(p for p in round2_dir.iterdir() if p.is_dir()):
        if (d / "per_slice_dice.jsonl").is_file():
            runs[d.name] = d
    return runs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _pct_linear(sorted_vals: list[float], q: float) -> float:
    """NumPy-default linear-interpolation percentile on a sorted list."""
    if not sorted_vals:
        raise ValueError("empty sample")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = q * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """95% percentile-bootstrap CI of the mean; deterministic for a fixed seed."""
    n = len(values)
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    try:
        import numpy as np

        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_resamples, n))
        means = np.asarray(values, dtype=float)[idx].mean(axis=1)
        lo = float(np.percentile(means, 100.0 * alpha / 2.0))
        hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
        return lo, hi
    except ImportError:
        rng = random.Random(seed)
        means = sorted(
            sum(values[i] for i in rng.choices(range(n), k=n)) / n
            for _ in range(n_resamples)
        )
        return _pct_linear(means, alpha / 2.0), _pct_linear(means, 1.0 - alpha / 2.0)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

def stratum_rows(
    run_label: str,
    records: list[dict],
    batch_mean: float | None,
    n_resamples: int,
    seed: int,
    fold: int | None = None,
) -> list[dict]:
    """One row per stratum (overall / positive_only / negative_only)."""
    strata = [
        ("overall", lambda r: True),
        ("positive_only", lambda r: not r["empty_gt"]),
        ("negative_only", lambda r: r["empty_gt"]),
    ]
    rows = []
    for name, sel in strata:
        vals = [r["dice"] for r in records if sel(r)]
        if not vals:
            continue
        est = sum(vals) / len(vals)
        lo, hi = bootstrap_ci(vals, n_resamples=n_resamples, seed=seed)
        rows.append(
            {
                "run_label": run_label,
                "stratum": name,
                "n": len(vals),
                "point_estimate": est,
                "ci_low": lo,
                "ci_high": hi,
                "ci_width": hi - lo,
                "batch_mean_estimate": batch_mean,
                "batch_mean_available": batch_mean is not None,
                "fold": fold,
                "fold_sd": None,
            }
        )
    return rows


def across_folds_row(
    base: str,
    fold_rows: dict[str, list[dict]],
    n_resamples: int,
    seed: int,
) -> dict:
    """Mean +/- SD of per-fold overall point estimates; CI over fold means."""
    fold_means: list[tuple[int, float]] = []
    total_n = 0
    for label in sorted(fold_rows):
        overall = next(r for r in fold_rows[label] if r["stratum"] == "overall")
        fold_means.append((overall["n"], overall["point_estimate"]))
        total_n += overall["n"]
    means = [m for _, m in fold_means]
    est = sum(means) / len(means)
    sd = sample_sd(means)
    lo, hi = bootstrap_ci(means, n_resamples=n_resamples, seed=seed)
    return {
        "run_label": base,
        "stratum": "across_folds",
        "n": total_n,
        "point_estimate": est,
        "ci_low": lo,
        "ci_high": hi,
        "ci_width": hi - lo,
        "batch_mean_estimate": None,
        "batch_mean_available": False,
        "fold": None,
        "fold_sd": sd,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round2-dir", type=Path, default=DEFAULT_ROUND2,
                        help="directory containing <run_label>/per_slice_dice.jsonl dumps")
    parser.add_argument("--out", type=Path, default=None,
                        help="output CSV (default: <round2-dir>/dice_ci_summary.csv)")
    parser.add_argument("--fold-aware", action="store_true",
                        help="group <base>_fold<N>of5 dumps and emit per-fold rows "
                             "plus a mean +/- SD across-folds row")
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = args.out or (args.round2_dir / "dice_ci_summary.csv")

    runs = discover_runs(args.round2_dir)
    if not runs:
        raise SystemExit(f"no per_slice_dice.jsonl dumps found under {args.round2_dir}")

    all_rows: list[dict] = []
    for label, run_dir in runs.items():
        records = load_records(run_dir / "per_slice_dice.jsonl")
        batch_mean = batch_mean_from_summary(run_dir)
        fold = None
        m = FOLD_RE.match(label)
        if m:
            fold = int(m.group("n"))
        all_rows.extend(stratum_rows(label, records, batch_mean, args.n_resamples, args.seed, fold=fold))

    if args.fold_aware:
        # Group fold dumps by base; emit across_folds rows for bases with >=2 folds.
        bases: dict[str, list[str]] = {}
        for label in runs:
            m = FOLD_RE.match(label)
            if m:
                bases.setdefault(m.group("base"), []).append(label)
        for base, fold_labels in sorted(bases.items()):
            if len(fold_labels) < 2:
                continue
            fold_rows = {
                lbl: [r for r in all_rows if r["run_label"] == lbl] for lbl in fold_labels
            }
            all_rows.append(across_folds_row(base, fold_rows, args.n_resamples, args.seed))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            out = dict(row)
            out["point_estimate"] = f"{row['point_estimate']:.6f}"
            out["ci_low"] = f"{row['ci_low']:.6f}"
            out["ci_high"] = f"{row['ci_high']:.6f}"
            out["ci_width"] = f"{row['ci_width']:.6f}"
            out["batch_mean_estimate"] = (
                f"{row['batch_mean_estimate']:.6f}" if row["batch_mean_available"] else ""
            )
            out["batch_mean_available"] = str(row["batch_mean_available"]).lower()
            out["fold"] = "" if row["fold"] is None else row["fold"]
            out["fold_sd"] = f"{row['fold_sd']:.6f}" if row["fold_sd"] is not None else ""
            writer.writerow(out)

    # Stdout table.
    print(f"{'run_label':<28} {'stratum':<14} {'n':>5}  {'point_est':>10}  "
          f"{'95% CI':>20}  {'width':>8}  {'batch_mean':>10}  {'fold_sd':>8}")
    for row in all_rows:
        bm = f"{row['batch_mean_estimate']:.6f}" if row["batch_mean_available"] else "-"
        fsd = f"{row['fold_sd']:.6f}" if row["fold_sd"] is not None else "-"
        print(f"{row['run_label']:<28} {row['stratum']:<14} {row['n']:>5}  "
              f"{row['point_estimate']:>10.6f}  "
              f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}]  "
              f"{row['ci_width']:>8.6f}  {bm:>10}  {fsd:>8}")
    print(f"\nWrote {out_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
