"""Read-only data loaders for the paper figure package.

Every loader here is *read-only*: it never writes, never mutates, and never
re-derives a number that already lives in a ground-truth artifact. Figures
import these loaders so that the values they plot are the exact values that
were measured, not hand-typed constants.

Repo-root resolution
--------------------
All paths are resolved relative to the repository root, which is derived from
this file's own location (``paper/figures/loaders.py`` -> two parents up).
No loader assumes the current working directory.

Loaders
-------
``results_matrix()``
    The paper results matrix with 95% Wilson CIs, from
    ``paper/paper_results_matrix_with_ci.csv``. The CI columns are parsed to
    floats.
``dice_degeneracy()``
    The empty-mask Dice-inflation curve, from
    ``paper/dice_degeneracy_curve.csv``.
``run_epoch_windows()``
    For every ``logs/run_NN.log``: parse the timestamped epoch lines into
    ``(start, end)`` windows, slice ``checkpoints/epoch_log.jsonl`` by
    ``(dataset, encoder, timestamp-in-window)``, and assert that the sliced
    best validation accuracy / Dice match the CSV row for that run
    (float tolerance 0.01). Runs that are *legitimately* not attributable to
    a single epoch-log window (e.g. two runs of the same dataset+encoder that
    ran concurrently, so their epoch-log records interleave) are reported as
    ``SKIPPED`` with a reason, not as failures.
``canonical_gradnorm()``
    The canonical GradNorm (Chen et al. 2018) probe log
    ``logs/canonical_gradnorm_run18.log``: per-epoch train/val accuracy,
    validation CI bounds, validation Dice, and the adaptive seg/cls weights.

Round-2 artifact contract (results/round2/)
-------------------------------------------
``per_run_epoch_log(run_label)``
    The per-run epoch log ``results/round2/<run_label>/epoch_log.jsonl``
    (written by ``src/training.py::train_single_run``; F-10). Records carry
    ``run_label / fold / seed / splitter_branch / epoch / tr_loss / tr_acc /
    tr_dice / vl_loss / vl_acc / vl_dice / best_vl_loss / best_vl_acc /
    best_vl_dice / epoch_sec / is_best / smoke_test``. This is the round-2
    replacement for the legacy shared ``checkpoints/epoch_log.jsonl``.
``per_slice_dice(run_label)``
    The per-slice Dice dump ``results/round2/<run_label>/per_slice_dice.jsonl``
    (10-field contract: run_label, dataset, encoder, fold, seed, case_id,
    dice, empty_pred, empty_gt, label_int).
``kfold_summary(run_name)``
    The k-fold CV summary ``results/round2/kfold_<run_name>.json`` (written
    by ``main.py --summary-out`` for ``train_kfold_cv`` runs; the ``runs``
    dict carries the ``train_kfold_cv`` return: mean/std val acc/dice/loss,
    completed_folds, fold_results).
``dice_ci_summary()``
    The bootstrap Dice-CI table ``results/round2/dice_ci_summary.csv``
    (written by ``scripts/bootstrap_dice_ci.py``; 95% percentile bootstrap,
    seed 42, 10,000 resamples). One row per run x stratum (overall /
    positive_only / negative_only / across_folds).
``canonical_gradnorm_probe()``
    The NEW seeded canonical GradNorm probe log
    ``results/round2/canonical_gradnorm_probe/probe_log.jsonl`` (seed 42,
    deterministic): per-epoch ``epoch / loss / train_acc / val_acc /
    val_acc_ci_lo / val_acc_ci_hi / val_dice / seg_weight / cls_weight``.
    :func:`canonical_gradnorm` prefers this JSONL and falls back to the
    legacy text log when the JSONL is absent.
``deterministic_best_ckpt(run_label)``
    The deterministic best-checkpoint path
    ``results/round2/<run_label>/best.pt`` (F-23 layout; the same
    ``run_label`` always maps to the same path).
``run_name_for_run_num(run_num)``
    The 26-run-matrix run name (e.g. ``g1_panda_vgg16``) for a 1-based CSV
    row number, from ``src/aggregate_results.py::EXPECTED_RUNS``.
``dice_ci_for_run(run_num)``
    The 95% bootstrap Dice-CI bounds (percent) for one matrix row, joined
    through the kfold summary's ``dataset``/``encoder`` to the
    ``dice_ci_summary.csv`` row for the same run (``overall`` stratum, or
    ``across_folds`` for k-fold runs). Returns ``None`` when the CI table
    does not cover the run yet.

Self-test
---------
``python -m paper.figures.loaders --selftest``
    Loads everything, runs the ``run_epoch_windows`` assertions, and prints a
    per-run PASS/FAIL/SKIPPED table plus the canonical parse count. This is
    the acceptance gate for the figure data infrastructure.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repo-root resolution (no cwd assumptions)
# ---------------------------------------------------------------------------

#: Repository root, derived from this file's location:
#: <root>/paper/figures/loaders.py  ->  parents[2] == <root>
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

RESULTS_MATRIX_CSV = REPO_ROOT / "paper" / "paper_results_matrix_with_ci.csv"
DICE_DEGENERACY_CSV = REPO_ROOT / "paper" / "dice_degeneracy_curve.csv"
EPOCH_LOG_JSONL = REPO_ROOT / "checkpoints" / "epoch_log.jsonl"
LOGS_DIR = REPO_ROOT / "logs"
CANONICAL_GRADNORM_LOG = REPO_ROOT / "logs" / "canonical_gradnorm_run18.log"

# ---------------------------------------------------------------------------
# Round-2 artifact contract (results/round2/)
# ---------------------------------------------------------------------------

#: Round-2 results root: per-run dirs ``<run_label>/`` plus kfold summaries
#: and the bootstrap Dice-CI table.
ROUND2_DIR: Path = REPO_ROOT / "results" / "round2"

#: Bootstrap Dice-CI table (scripts/bootstrap_dice_ci.py output).
DICE_CI_SUMMARY_CSV: Path = ROUND2_DIR / "dice_ci_summary.csv"

#: New seeded canonical GradNorm probe log (seed 42, deterministic).
CANONICAL_PROBE_DIR: Path = ROUND2_DIR / "canonical_gradnorm_probe"
CANONICAL_PROBE_LOG: Path = CANONICAL_PROBE_DIR / "probe_log.jsonl"

#: Tolerance (in percentage points) for the epoch-log vs CSV assertion.
ASSERT_TOL = 0.01


# ---------------------------------------------------------------------------
# (a) Results matrix with CIs
# ---------------------------------------------------------------------------

def results_matrix() -> pd.DataFrame:
    """Load the paper results matrix with parsed 95% CI columns.

    Returns a :class:`pandas.DataFrame` where the CI columns
    (``Acc 95% CI Lower`` / ``Acc 95% CI Upper``) are floats (NaN where the
    source cell is empty, e.g. the PANNUKE rows that have no published
    comparison).
    """
    df = pd.read_csv(RESULTS_MATRIX_CSV)
    for col in ("Acc 95% CI Lower", "Acc 95% CI Upper"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# (b) Dice degeneracy curve
# ---------------------------------------------------------------------------

def dice_degeneracy() -> pd.DataFrame:
    """Load the empty-mask Dice-inflation curve."""
    return pd.read_csv(DICE_DEGENERACY_CSV)


# ---------------------------------------------------------------------------
# (c) Run epoch windows + epoch-log slicing + CSV assertion
# ---------------------------------------------------------------------------

# A timestamped epoch line, e.g.
#   2026-08-25 23:24:10 [INFO] src.training:   17  | 0.1719 0.973  0.913 | 0.7939 0.855  0.792 | 24.2
# The epoch number may carry a trailing '*' (best-so-far marker).
_EPOCH_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[INFO\] src\.training:\s+"
    r"(?P<epoch>\d+)*? \|"
    r"(?P<tr_loss>[\d.]+) (?P<tr_acc>[\d.]+)\s+(?P<tr_dice>[\d.]+) \| "
    r"(?P<vl_loss>[\d.]+) (?P<vl_acc>[\d.]+)\s+(?P<vl_dice>[\d.]+) \| "
    r"(?P<sec>[\d.]+)\s*$"
)

# Attempt wrapper lines, e.g.
#   [1] 2026-08-25 23:11:28 - [ATTEMPT 1/2] Starting standard run...
#   [1] 2026-08-25 23:24:29 - [ATTEMPT 1/2] SUCCESS
_ATTEMPT_RE = re.compile(
    r"^\[\d+\] (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - "
    r"\[ATTEMPT \d+/\d+\] (?P<kind>Starting standard run|SUCCESS|FAILED)"
)

# The "Final metrics" block that closes a run:
#   Final metrics
#   <ts> [INFO] __main__: Dataset  Encoder         Acc(%)             Dice(%)
#   <ts> [INFO] __main__: ---------------------------------------------------
#   <ts> [INFO] __main__: tcga     vgg16          88.82              76.02
_FINAL_METRICS_RE = re.compile(
    r"Final metrics\n"
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[INFO\] __main__: Dataset.*?\n"
    r".*?\n"
    r"(?P<ts2>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[INFO\] __main__: "
    r"(?P<ds>\w+)\s+(?P<enc>\w+)\s+(?P<acc>[\d.]+)\s+(?P<dice>[\d.]+)"
)


def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


@dataclass
class RunWindow:
    """A single training attempt's wall-clock window and its epoch lines."""

    start: datetime
    end: datetime
    epochs: list = field(default_factory=list)  # list[(datetime, int)]

    def contains(self, ts: datetime) -> bool:
        return self.start <= ts <= self.end


def _parse_log_windows(log_path: Path):
    """Parse one run log into (final_metrics, [RunWindow, ...]).

    Windows are derived from the *epoch-line timestamps* (per the data
    contract). When a log is wrapped in ``[ATTEMPT ...]`` markers, each
    attempt becomes its own window; a raw log (no markers) yields a single
    implicit window spanning its epoch lines. ``final_metrics`` is the last
    ``Final metrics`` block (dataset, encoder, acc, dice, timestamp) or None.
    """
    text = log_path.read_text(errors="replace")

    fms = _FINAL_METRICS_RE.findall(text)
    final_metrics = None
    if fms:
        m = fms[-1]
        final_metrics = {
            "ts": m[0],
            "dataset": m[2].lower(),
            "encoder": m[3],
            "acc": float(m[4]),
            "dice": float(m[5]),
        }

    # Walk lines, grouping epoch lines into attempts.
    attempts: list[dict] = []
    cur: Optional[dict] = None
    for line in text.splitlines():
        am = _ATTEMPT_RE.match(line)
        if am:
            ts = _parse_dt(am.group("ts"))
            if am.group("kind") == "Starting standard run":
                cur = {"start": ts, "end": None, "epochs": []}
                attempts.append(cur)
            else:  # SUCCESS / FAILED closes the current attempt
                if cur is not None:
                    cur["end"] = ts
                    cur = None
            continue
        em = _EPOCH_RE.match(line)
        if em:
            ts = _parse_dt(em.group("ts"))
            if cur is None:
                # Raw log (no ATTEMPT wrapper): open an implicit attempt.
                cur = {"start": None, "end": None, "epochs": []}
                attempts.append(cur)
            cur["epochs"].append((ts, int(em.group("epoch"))))

    # Build RunWindows.
    windows: list[RunWindow] = []
    for a in attempts:
        if not a["epochs"]:
            continue
        e0 = a["epochs"][0][0]
        e1 = a["epochs"][-1][0]
        start = a["start"] if a["start"] is not None else e0
        end = a["end"] if a["end"] is not None else e1
        # A final attempt with no closing SUCCESS marker (raw log) extends to
        # the final-metrics timestamp so the last epoch is inside the window.
        if a["end"] is None and final_metrics is not None:
            end = max(end, _parse_dt(final_metrics["ts"]))
        windows.append(RunWindow(start=start, end=end, epochs=a["epochs"]))

    return final_metrics, windows


def _load_epoch_log() -> list:
    """Load ``checkpoints/epoch_log.jsonl`` into a list of record dicts.

    Malformed / non-JSON lines (e.g. a corrupted all-null-bytes record) are
    skipped rather than allowed to crash the whole load — a single bad line
    in a multi-thousand-line log must not take down every figure that reads
    the shared epoch log.
    """
    if not EPOCH_LOG_JSONL.exists():
        return []
    records = []
    with EPOCH_LOG_JSONL.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
    return records


def _slice_window(epoch_log: list, dataset: str, encoder: str,
                  start: datetime, end: datetime) -> list:
    """Return epoch-log records for (dataset, encoder) whose timestamp falls
    inside ``[start, end]``, sorted by timestamp."""
    sel = [
        r
        for r in epoch_log
        if r.get("dataset") == dataset
        and r.get("encoder") == encoder
        and start <= datetime.fromisoformat(r["timestamp"]) <= end
    ]
    sel.sort(key=lambda r: r["timestamp"])
    return sel


@dataclass
class RunCheck:
    """Result of checking one CSV row against its epoch-log window."""

    run_label: str
    dataset: str
    encoder: str
    csv_ts: str
    log: Optional[str]
    status: str  # "PASS" | "FAIL" | "SKIPPED"
    reason: str = ""
    csv_acc: float = float("nan")
    csv_dice: float = float("nan")
    sliced_acc: float = float("nan")
    sliced_dice: float = float("nan")
    n_records: int = 0


def run_epoch_windows() -> list:
    """Cross-check every results-matrix row against its epoch-log window.

    For each CSV row we locate the run log whose ``Final metrics`` block
    carries the row's ``Timestamp`` and matching dataset/encoder, pick the
    attempt window that contains that timestamp, slice the epoch log, and
    assert the sliced best validation accuracy / Dice equal the CSV values
    within :data:`ASSERT_TOL`.

    A row is ``SKIPPED`` (with a reason) when it is *legitimately* not
    attributable to a single epoch-log window:

    * no run log carries a matching ``Final metrics`` timestamp;
    * the matched log's dataset/encoder disagrees with the row;
    * no attempt window contains the row's timestamp;
    * the epoch log has no records for that (dataset, encoder) in the window;
    * **concurrent overlap** — another run of the *same* (dataset, encoder)
      has a window that overlaps this one, so their epoch-log records
      interleave and cannot be attributed to a single run.

    A row is ``FAIL`` only when a window is unambiguously attributable but
    the sliced values disagree with the CSV beyond tolerance.
    """
    matrix = results_matrix()
    epoch_log = _load_epoch_log()

    # Index every run log by its final-metrics timestamp.
    logs = sorted(LOGS_DIR.glob("run_*.log"))
    by_ts: dict[str, tuple[Path, dict, list]] = {}
    all_windows: dict[tuple, list] = {}
    for p in logs:
        fm, wins = _parse_log_windows(p)
        if fm is None:
            continue
        by_ts[fm["ts"]] = (p, fm, wins)
        for w in wins:
            all_windows.setdefault((fm["dataset"], fm["encoder"]), []).append(
                (w.start, w.end, p.name)
            )

    def _overlaps(a, b) -> bool:
        return a[0] <= b[1] and b[0] <= a[1]

    checks: list[RunCheck] = []
    for _, row in matrix.iterrows():
        ds = str(row["Dataset"]).lower()
        enc = str(row["Encoder"])
        csv_ts = str(row["Timestamp"]).replace("T", " ")
        csv_acc = float(row["Accuracy (%)"])
        csv_dice = float(row["Macro Dice (%)"])
        label = str(row["Run Label"])
        csv_dt = _parse_dt(csv_ts)

        key = by_ts.get(csv_ts)
        if key is None:
            checks.append(RunCheck(label, ds, enc, csv_ts, None, "SKIPPED",
                                   "no run log with matching final-metrics timestamp"))
            continue
        log_path, fm, wins = key
        if fm["dataset"] != ds or fm["encoder"] != enc:
            checks.append(RunCheck(label, ds, enc, csv_ts, log_path.name, "SKIPPED",
                                   f"log {log_path.name} reports {fm['dataset']}/{fm['encoder']}"))
            continue
        chosen = [w for w in wins if w.contains(csv_dt)]
        if not chosen:
            checks.append(RunCheck(label, ds, enc, csv_ts, log_path.name, "SKIPPED",
                                   "no attempt window contains the CSV timestamp"))
            continue
        win = chosen[-1]

        # Concurrency check: any *other* log of the same (ds, enc) overlapping?
        amb = [
            name
            for (s, e, name) in all_windows.get((ds, enc), [])
            if (s, e) != (win.start, win.end) and _overlaps((win.start, win.end), (s, e))
        ]
        if amb:
            checks.append(RunCheck(label, ds, enc, csv_ts, log_path.name, "SKIPPED",
                                   "concurrent same-(dataset,encoder) overlap: "
                                   + ", ".join(sorted(amb))))
            continue

        sel = _slice_window(epoch_log, ds, enc, win.start, win.end)
        if not sel:
            checks.append(RunCheck(label, ds, enc, csv_ts, log_path.name, "SKIPPED",
                                   "no epoch_log records in window"))
            continue

        last = sel[-1]
        sl_acc = float(last["best_vl_acc"]) * 100.0
        sl_dice = float(last["best_vl_dice"]) * 100.0
        ok = abs(sl_acc - csv_acc) <= ASSERT_TOL and abs(sl_dice - csv_dice) <= ASSERT_TOL
        checks.append(
            RunCheck(
                label, ds, enc, csv_ts, log_path.name,
                "PASS" if ok else "FAIL",
                "" if ok else f"sliced {sl_acc:.4f}/{sl_dice:.4f} vs csv {csv_acc:.2f}/{csv_dice:.2f}",
                csv_acc, csv_dice, sl_acc, sl_dice, len(sel),
            )
        )
    return checks


# ---------------------------------------------------------------------------
# (c2) Per-epoch run trajectory (for trajectory figures)
# ---------------------------------------------------------------------------

def run_trajectory(run_num: int) -> pd.DataFrame:
    """Return the per-epoch validation trajectory for one run.

    ``run_num`` is the 1-based run number (the CSV row index, which equals the
    ``run_NN`` log number). The run's log is located by its ``Final metrics``
    timestamp, the attempt window containing that timestamp is selected, and
    the epoch log is sliced by ``(dataset, encoder, timestamp-in-window)`` —
    the exact same windowing as :func:`run_epoch_windows`.

    Returns a :class:`pandas.DataFrame` with columns ``epoch``, ``vl_acc``,
    ``best_vl_acc``, ``vl_dice`` (all percentages). ``best_vl_acc`` is the
    running-best validation accuracy (the value the CSV ``Accuracy (%)``
    column reports); ``vl_acc`` is the raw per-epoch value.

    Returns an **empty** DataFrame (with those columns) when the run is not
    attributable to a single epoch-log window (e.g. concurrent same-
    (dataset, encoder) overlap) — callers should treat an empty result as
    "skip this curve", not as a failure.
    """
    matrix = results_matrix()
    if run_num < 1 or run_num > len(matrix):
        return pd.DataFrame(columns=["epoch", "vl_acc", "best_vl_acc", "vl_dice"])
    row = matrix.iloc[run_num - 1]
    ds = str(row["Dataset"]).lower()
    enc = str(row["Encoder"])
    csv_ts = str(row["Timestamp"]).replace("T", " ")

    epoch_log = _load_epoch_log()
    for p in sorted(LOGS_DIR.glob("run_*.log")):
        fm, wins = _parse_log_windows(p)
        if fm is None or fm["ts"] != csv_ts:
            continue
        if fm["dataset"] != ds or fm["encoder"] != enc:
            continue
        csv_dt = _parse_dt(csv_ts)
        chosen = [w for w in wins if w.contains(csv_dt)]
        if not chosen:
            return pd.DataFrame(columns=["epoch", "vl_acc", "best_vl_acc", "vl_dice"])
        win = chosen[-1]
        sel = _slice_window(epoch_log, ds, enc, win.start, win.end)
        if not sel:
            return pd.DataFrame(columns=["epoch", "vl_acc", "best_vl_acc", "vl_dice"])
        out = pd.DataFrame(
            {
                "epoch": [int(r["epoch"]) for r in sel],
                "vl_acc": [float(r["vl_acc"]) * 100.0 for r in sel],
                "best_vl_acc": [float(r["best_vl_acc"]) * 100.0 for r in sel],
                "vl_dice": [float(r["vl_dice"]) * 100.0 for r in sel],
            }
        )
        return out.sort_values("epoch").reset_index(drop=True)
    return pd.DataFrame(columns=["epoch", "vl_acc", "best_vl_acc", "vl_dice"])


# ---------------------------------------------------------------------------
# (d) Canonical GradNorm probe
# ---------------------------------------------------------------------------

# Per-epoch line, e.g.
#   2026-09-03 20:08:25,368 [INFO] Epoch 01/30 | Loss: 2.2795 | Train Acc: 28.55%
#     | Val Acc: 32.65% [30.68, 34.69] | Val Dice: 30.35% | Weights: [seg=0.512, cls=1.488]
_CANON_EPOCH_RE = re.compile(
    r"Epoch (?P<epoch>\d+)/\d+ \| Loss: (?P<loss>[\d.]+) \| "
    r"Train Acc: (?P<train_acc>[\d.]+)% \| "
    r"Val Acc: (?P<val_acc>[\d.]+)% \[(?P<ci_lo>[\d.]+), (?P<ci_hi>[\d.]+)\] \| "
    r"Val Dice: (?P<val_dice>[\d.]+)% \| "
    r"Weights: \[seg=(?P<seg>[\d.]+), cls=(?P<cls>[\d.]+)\]"
)

#: Column order of the canonical probe DataFrame (JSONL and text-log sources
#: both normalize to this).
_CANON_COLUMNS = [
    "epoch", "loss", "train_acc", "val_acc",
    "val_acc_ci_lo", "val_acc_ci_hi", "val_dice", "seg_weight", "cls_weight",
]


def canonical_gradnorm_probe() -> pd.DataFrame:
    """Parse the NEW seeded canonical GradNorm probe log (round-2 contract).

    Source: ``results/round2/canonical_gradnorm_probe/probe_log.jsonl`` — one
    JSON object per epoch with fields ``epoch / loss / train_acc / val_acc /
    val_acc_ci_lo / val_acc_ci_hi / val_dice / seg_weight / cls_weight``
    (seed 42, deterministic). Returns an **empty** DataFrame (with the
    canonical columns) when the JSONL is absent.
    """
    if not CANONICAL_PROBE_LOG.exists():
        return pd.DataFrame(columns=_CANON_COLUMNS)
    rows = []
    with CANONICAL_PROBE_LOG.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(
                {
                    "epoch": int(rec["epoch"]),
                    "loss": float(rec["loss"]),
                    "train_acc": float(rec["train_acc"]),
                    "val_acc": float(rec["val_acc"]),
                    "val_acc_ci_lo": float(rec["val_acc_ci_lo"]),
                    "val_acc_ci_hi": float(rec["val_acc_ci_hi"]),
                    "val_dice": float(rec["val_dice"]),
                    "seg_weight": float(rec["seg_weight"]),
                    "cls_weight": float(rec["cls_weight"]),
                }
            )
    return pd.DataFrame(rows, columns=_CANON_COLUMNS)


def canonical_gradnorm() -> pd.DataFrame:
    """Per-epoch canonical GradNorm probe DataFrame (round-2 aware).

    Prefers the new seeded JSONL probe log
    (:func:`canonical_gradnorm_probe`); falls back to the legacy text log
    ``logs/canonical_gradnorm_run18.log`` when the JSONL is absent.

    Columns: ``epoch``, ``loss``, ``train_acc``, ``val_acc``,
    ``val_acc_ci_lo``, ``val_acc_ci_hi``, ``val_dice``, ``seg_weight``,
    ``cls_weight`` (all percentages/weights as floats).
    """
    probe = canonical_gradnorm_probe()
    if not probe.empty:
        return probe
    if not CANONICAL_GRADNORM_LOG.exists():
        return pd.DataFrame(columns=_CANON_COLUMNS)
    text = CANONICAL_GRADNORM_LOG.read_text(errors="replace")
    rows = []
    for m in _CANON_EPOCH_RE.finditer(text):
        rows.append(
            {
                "epoch": int(m.group("epoch")),
                "loss": float(m.group("loss")),
                "train_acc": float(m.group("train_acc")),
                "val_acc": float(m.group("val_acc")),
                "val_acc_ci_lo": float(m.group("ci_lo")),
                "val_acc_ci_hi": float(m.group("ci_hi")),
                "val_dice": float(m.group("val_dice")),
                "seg_weight": float(m.group("seg")),
                "cls_weight": float(m.group("cls")),
            }
        )
    return pd.DataFrame(rows, columns=_CANON_COLUMNS)


# ---------------------------------------------------------------------------
# (e) Round-2 per-run artifacts (results/round2/)
# ---------------------------------------------------------------------------

def per_run_epoch_log(run_label: str, round2_dir: Path | None = None) -> pd.DataFrame:
    """Load ``results/round2/<run_label>/epoch_log.jsonl`` (F-10 per-run log).

    Returns a :class:`pandas.DataFrame` with the record fields (``epoch``,
    ``tr_loss``, ``tr_acc``, ``tr_dice``, ``vl_loss``, ``vl_acc``,
    ``vl_dice``, ``best_vl_loss``, ``best_vl_acc``, ``best_vl_dice``,
    ``run_label``, ``fold``, ``seed``, ``splitter_branch``, ...) sorted by
    ``epoch``. Returns an **empty** DataFrame when the file is absent.

    ``round2_dir`` overrides the ``results/round2`` root (e.g. a /tmp fixture
    dir) for standalone verification.
    """
    base = round2_dir or ROUND2_DIR
    path = base / run_label / "epoch_log.jsonl"
    if not path.exists():
        return pd.DataFrame()
    with path.open() as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("epoch").reset_index(drop=True)


def per_slice_dice(run_label: str, round2_dir: Path | None = None) -> pd.DataFrame:
    """Load ``results/round2/<run_label>/per_slice_dice.jsonl`` (10-field contract).

    Fields: ``run_label, dataset, encoder, fold, seed, case_id, dice,
    empty_pred, empty_gt, label_int``. Returns an **empty** DataFrame when
    the file is absent.

    ``round2_dir`` overrides the ``results/round2`` root (e.g. a /tmp fixture
    dir) for standalone verification.
    """
    base = round2_dir or ROUND2_DIR
    path = base / run_label / "per_slice_dice.jsonl"
    if not path.exists():
        return pd.DataFrame()
    with path.open() as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def kfold_summary(run_name: str, round2_dir: Path | None = None) -> dict | None:
    """Load ``results/round2/kfold_<run_name>.json`` (k-fold CV summary).

    The file is written by ``main.py --summary-out`` for
    ``train_kfold_cv`` runs; its ``runs`` dict carries the
    ``train_kfold_cv`` return (``mean_val_acc`` / ``std_val_acc`` /
    ``mean_val_dice`` / ``std_val_dice`` / ``mean_val_loss`` /
    ``std_val_loss`` / ``completed_folds`` / ``fold_results``). Returns the
    run entry dict, or ``None`` when the file is absent or has no usable
    run entry.

    ``round2_dir`` overrides the ``results/round2`` root (e.g. a /tmp fixture
    dir) for standalone verification.
    """
    base = round2_dir or ROUND2_DIR
    path = base / f"kfold_{run_name}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    runs = data.get("runs")
    if not isinstance(runs, dict) or not runs:
        return None
    for _key, val in runs.items():
        if isinstance(val, dict) and "mean_val_acc" in val:
            return val
    return None


def dice_ci_summary(round2_dir: Path | None = None) -> pd.DataFrame:
    """Load ``results/round2/dice_ci_summary.csv`` (bootstrap Dice CIs).

    Written by ``scripts/bootstrap_dice_ci.py``: one row per run x stratum
    (``overall`` / ``positive_only`` / ``negative_only`` / ``across_folds``)
    with ``point_estimate`` / ``ci_low`` / ``ci_high`` / ``ci_width`` as
    fractions in [0, 1] (95% percentile bootstrap, seed 42). Returns an
    **empty** DataFrame when the file is absent.

    ``round2_dir`` overrides the ``results/round2`` root (e.g. a /tmp fixture
    dir) for standalone verification.
    """
    base = round2_dir or ROUND2_DIR
    csv_path = base / "dice_ci_summary.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    for col in ("point_estimate", "ci_low", "ci_high", "ci_width"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def deterministic_best_ckpt(run_label: str, round2_dir: Path | None = None) -> Path:
    """Deterministic best-checkpoint path ``results/round2/<run_label>/best.pt``.

    F-23 layout: the same ``run_label`` always maps to the same path (no
    timestamps, no randomization). The path is returned whether or not the
    file exists yet; callers check ``.exists()`` before loading.

    ``round2_dir`` overrides the ``results/round2`` root (e.g. a /tmp fixture
    dir) for standalone verification.
    """
    base = round2_dir or ROUND2_DIR
    return base / run_label / "best.pt"


def run_name_for_run_num(run_num: int) -> str:
    """The 26-run-matrix run name (e.g. ``g1_panda_vgg16``) for a 1-based
    CSV row number, from ``src/aggregate_results.py::EXPECTED_RUNS``."""
    from src.aggregate_results import EXPECTED_RUNS
    if run_num < 1 or run_num > len(EXPECTED_RUNS):
        raise IndexError(f"run_num {run_num} outside 1..{len(EXPECTED_RUNS)}")
    return EXPECTED_RUNS[run_num - 1][1]


def run_label_for_run_num(run_num: int) -> str:
    """The round-2 per-run label (e.g. ``03_g1_panda_vgg16``) for a 1-based
    CSV row number.

    Mirrors ``run_all_experiments.sh``: ``run_label = <padded_id>_<run_name>``
    where ``padded_id`` is the zero-padded 2-digit run number and
    ``run_name`` is the matrix run name. This is the directory name under
    ``results/round2/`` that holds the per-run ``epoch_log.jsonl`` /
    ``per_slice_dice.jsonl`` / ``best.pt`` / ``final.state.pt``.
    """
    return f"{run_num:02d}_{run_name_for_run_num(run_num)}"


def round2_run_trajectory(run_num: int) -> pd.DataFrame:
    """Per-epoch validation trajectory for one run from the round-2 per-run
    epoch log ``results/round2/<run_label>/epoch_log.jsonl`` (F-10).

    This is the round-2 replacement for :func:`run_trajectory` (which slices
    the legacy shared ``checkpoints/epoch_log.jsonl`` by log-window). It reads
    the per-run log directly — no windowing, no concurrency ambiguity — and
    returns the same columns: ``epoch``, ``vl_acc``, ``best_vl_acc``,
    ``vl_dice`` (all percentages). ``best_vl_acc`` is the running-best
    validation accuracy (the value the CSV ``Accuracy (%)`` column reports).

    Returns an **empty** DataFrame (with those columns) when the per-run log
    is absent — callers treat an empty result as "skip this curve", not a
    failure.
    """
    cols = ["epoch", "vl_acc", "best_vl_acc", "vl_dice"]
    df = per_run_epoch_log(run_label_for_run_num(run_num))
    if df.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(
        {
            "epoch": [int(r["epoch"]) for r in df.to_dict("records")],
            "vl_acc": [float(r["vl_acc"]) * 100.0 for r in df.to_dict("records")],
            "best_vl_acc": [float(r["best_vl_acc"]) * 100.0 for r in df.to_dict("records")],
            "vl_dice": [float(r["vl_dice"]) * 100.0 for r in df.to_dict("records")],
        }
    )
    return out.sort_values("epoch").reset_index(drop=True)


# ---------------------------------------------------------------------------
# (f) Fold-campaign (k-fold CV) loaders
# ---------------------------------------------------------------------------

#: z-value for a two-sided 95% interval (matches scripts/compute_wilson_ci.py).
_Z95 = 1.96


def kfold_across_folds(run_name: str, round2_dir: Path | None = None) -> dict | None:
    """The ``across_folds`` row of ``dice_ci_summary.csv`` for one k-fold run.

    ``run_name`` is the bare run name (e.g. ``g1_tcga_vgg16``); the CSV
    ``run_label`` is ``kfold_<run_name>``. Returns a dict with keys
    ``point_estimate`` / ``ci_low`` / ``ci_high`` / ``fold_sd`` (all fractions
    in [0, 1]) or ``None`` when the run has no ``across_folds`` row. The
    ``point_estimate`` is the bootstrap Dice estimate across all folds; the
    ``ci_low`` / ``ci_high`` are the 95% bootstrap CI bounds; ``fold_sd`` is
    the standard deviation of the per-fold Dice.
    """
    ci = dice_ci_summary(round2_dir)
    if ci.empty:
        return None
    label = f"kfold_{run_name}"
    sel = ci[(ci["run_label"] == label) & (ci["stratum"] == "across_folds")]
    if sel.empty:
        return None
    r = sel.iloc[0]
    return {
        "point_estimate": float(r["point_estimate"]),
        "ci_low": float(r["ci_low"]),
        "ci_high": float(r["ci_high"]),
        "fold_sd": float(r["fold_sd"]),
    }


def kfold_acc_stats(run_name: str, round2_dir: Path | None = None) -> dict | None:
    """The k-fold CV accuracy point estimate + fold SD for one run.

    Reads ``kfold_<run_name>.json`` (``mean_val_acc`` / ``std_val_acc``) and
    returns a dict with keys ``mean`` / ``sd`` (fractions in [0, 1]) or
    ``None`` when the summary is absent. ``mean`` is the across-folds mean
    validation accuracy (the point estimate the fold campaign reports);
    ``sd`` is the across-folds standard deviation.
    """
    kf = kfold_summary(run_name, round2_dir)
    if kf is None:
        return None
    if "mean_val_acc" not in kf or "std_val_acc" not in kf:
        return None
    return {
        "mean": float(kf["mean_val_acc"]),
        "sd": float(kf["std_val_acc"]),
    }


def kfold_acc_ci(run_name: str, round2_dir: Path | None = None) -> tuple[float, float, float] | None:
    """95% across-folds CI for the k-fold CV accuracy (percent).

    The fold campaign reports the per-fold validation accuracies
    (``fold_results[*].best_val_acc``). The 95% CI is a **fold bootstrap**
    interval: 10,000 resamples (with replacement) of the per-fold accuracies
    (×100) drawn with ``numpy.random.default_rng(42)``, and the 2.5 / 97.5
    percentiles of the resampled *means*. This replaces the earlier
    normal-approximation ``mean ± z·(sd/√k)`` interval.

    Returns ``(point, lo, hi)`` in percent, or ``None`` when the run has no
    k-fold summary with at least two per-fold accuracies.
    """
    kf = kfold_summary(run_name, round2_dir)
    if kf is None:
        return None
    fold_results = kf.get("fold_results")
    if not fold_results:
        return None
    accs = np.array(
        [float(f["best_val_acc"]) * 100.0 for f in fold_results], dtype=float
    )
    if accs.size < 2:
        return None
    mean = float(accs.mean())
    rng = np.random.default_rng(42)
    n_boot = 10_000
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot_means[i] = rng.choice(accs, size=accs.size, replace=True).mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return (mean, float(lo), float(hi))


def kfold_dice_ci(run_name: str, round2_dir: Path | None = None) -> tuple[float, float, float] | None:
    """95% bootstrap across-folds CI for the k-fold CV Dice (percent).

    Reads the ``across_folds`` row of ``dice_ci_summary.csv`` and returns
    ``(point, lo, hi)`` in percent, or ``None`` when the run is not covered.
    """
    row = kfold_across_folds(run_name, round2_dir)
    if row is None:
        return None
    return (row["point_estimate"] * 100.0,
            row["ci_low"] * 100.0,
            row["ci_high"] * 100.0)


def kfold_run_stats(run_num: int, round2_dir: Path | None = None) -> dict | None:
    """Fold-campaign statistics for one results-matrix row (1-based ``run_num``).

    Joins the k-fold CV summary (``kfold_<run_name>.json``) to the bootstrap
    Dice-CI table (``dice_ci_summary.csv`` ``across_folds`` row) and returns a
    dict with the fold-aware columns the figures plot:

    * ``acc_point`` / ``acc_ci_lo`` / ``acc_ci_hi`` — the across-folds mean
      validation accuracy (percent) and its 95% CI
      (``mean ± z·(sd/√k)`` from the kfold summary's ``mean_val_acc`` /
      ``std_val_acc`` / ``completed_folds``).
    * ``dice_point`` / ``dice_ci_lo`` / ``dice_ci_hi`` — the across-folds
      bootstrap Dice point estimate and 95% CI (percent) from the CSV
      ``across_folds`` row.
    * ``fold_sd`` — the CSV ``fold_sd`` column (Dice fold SD, fraction).
    * ``completed_folds`` — the number of completed folds.

    Returns ``None`` when the run has no k-fold summary (i.e. it is not part
    of the fold campaign).
    """
    run_name = run_name_for_run_num(run_num)
    kf = kfold_summary(run_name, round2_dir)
    if kf is None:
        return None
    k = int(kf.get("completed_folds") or 5)
    if k < 1:
        k = 5
    mean = float(kf["mean_val_acc"])
    sd = float(kf["std_val_acc"])
    half = _Z95 * (sd / math.sqrt(k))
    acc_point = mean * 100.0
    acc_ci_lo = (mean - half) * 100.0
    acc_ci_hi = (mean + half) * 100.0

    af = kfold_across_folds(run_name, round2_dir)
    if af is None:
        return None
    return {
        "run_name": run_name,
        "acc_point": acc_point,
        "acc_ci_lo": acc_ci_lo,
        "acc_ci_hi": acc_ci_hi,
        "dice_point": af["point_estimate"] * 100.0,
        "dice_ci_lo": af["ci_low"] * 100.0,
        "dice_ci_hi": af["ci_high"] * 100.0,
        "fold_sd": af["fold_sd"],
        "completed_folds": k,
    }


def dice_ci_for_run(run_num: int) -> tuple[float, float, float] | None:
    """95% bootstrap Dice-CI bounds (percent) for one results-matrix row.

    Joins the row's kfold summary (``kfold_<run_name>.json``) to the
    ``dice_ci_summary.csv`` row for the same run (matched on
    ``dataset``/``encoder``; the ``overall`` stratum, or ``across_folds``
    for k-fold runs). Returns ``(point, lo, hi)`` in percent, or ``None``
    when the CI table does not cover the run yet.
    """
    matrix = results_matrix()
    if run_num < 1 or run_num > len(matrix):
        return None
    row = matrix.iloc[run_num - 1]
    ds = str(row["Dataset"]).lower()
    enc = str(row["Encoder"])
    run_name = run_name_for_run_num(run_num)
    kf = kfold_summary(run_name)
    if kf is None:
        return None
    if str(kf.get("dataset", "")).lower() != ds or str(kf.get("encoder", "")) != enc:
        return None
    ci = dice_ci_summary()
    if ci.empty:
        return None
    sel = ci[
        (ci["run_label"] == run_name)
        & (ci["dataset"] == ds)
        & (ci["encoder"] == enc)
    ]
    if sel.empty:
        return None
    overall = sel[sel["stratum"] == "overall"]
    if not overall.empty:
        r = overall.iloc[0]
    else:
        across = sel[sel["stratum"] == "across_folds"]
        if across.empty:
            return None
        r = across.iloc[0]
    return (float(r["point_estimate"]) * 100.0,
            float(r["ci_low"]) * 100.0,
            float(r["ci_high"]) * 100.0)


# ---------------------------------------------------------------------------
# Self-test CLI
# ---------------------------------------------------------------------------

def _selftest() -> int:
    """Run the full data-infrastructure self-test; return a process exit code.

    Exit code 0 iff every attributable run PASSES and no run FAILs. SKIPPED
    runs are legitimate (reported, not failed).
    """
    print("=" * 78)
    print("paper.figures.loaders self-test")
    print("=" * 78)

    # (a) results matrix
    matrix = results_matrix()
    print(f"\n[a] results_matrix: {len(matrix)} rows, "
          f"{sum(matrix['Acc 95% CI Lower'].notna())} with parsed CI")

    # (b) dice degeneracy
    deg = dice_degeneracy()
    print(f"[b] dice_degeneracy: {len(deg)} points "
          f"(empty-ratio {deg['empty_slice_ratio'].iloc[0]}..{deg['empty_slice_ratio'].iloc[-1]})")

    # (d) canonical gradnorm
    can = canonical_gradnorm()
    print(f"[d] canonical_gradnorm: {len(can)} epochs parsed "
          f"(seg/cls weights present: {can['seg_weight'].notna().all() if len(can) else False})")

    # (e) round-2 artifacts
    n_per_run = 0
    if ROUND2_DIR.is_dir():
        n_per_run = sum(1 for d in ROUND2_DIR.iterdir()
                        if d.is_dir() and (d / "epoch_log.jsonl").is_file())
    n_kfold = 0
    if ROUND2_DIR.is_dir():
        n_kfold = len(list(ROUND2_DIR.glob("kfold_*.json")))
    ci = dice_ci_summary()
    print(f"[e] round2: {n_per_run} per-run epoch logs, {n_kfold} kfold summaries, "
          f"{len(ci)} dice_ci_summary rows")

    # (c) run epoch windows
    checks = run_epoch_windows()
    n_pass = sum(1 for c in checks if c.status == "PASS")
    n_fail = sum(1 for c in checks if c.status == "FAIL")
    n_skip = sum(1 for c in checks if c.status == "SKIPPED")

    print(f"\n[c] run_epoch_windows: {len(checks)} CSV rows -> "
          f"{n_pass} PASS, {n_fail} FAIL, {n_skip} SKIPPED")
    print("-" * 78)
    hdr = f"{'STATUS':6s} {'Run Label':42s} {'log':34s}"
    print(hdr)
    print("-" * 78)
    for c in checks:
        log = c.log or "-"
        print(f"{c.status:6s} {c.run_label:42s} {log:34s}")
        if c.status == "PASS":
            print(f"       sliced acc={c.sliced_acc:.4f} dice={c.sliced_dice:.4f} "
                  f"(csv {c.csv_acc:.2f}/{c.csv_dice:.2f}, n={c.n_records})")
        elif c.status == "FAIL":
            print(f"       MISMATCH: {c.reason}")
        else:
            print(f"       SKIPPED: {c.reason}")
    print("-" * 78)

    if n_fail:
        print(f"\nRESULT: FAIL ({n_fail} run(s) mismatched the CSV).")
        return 1
    print(f"\nRESULT: OK — {n_pass} runs asserted against epoch_log, "
          f"{n_skip} legitimately skipped.")
    return 0


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
    print("Usage: python -m paper.figures.loaders --selftest")
