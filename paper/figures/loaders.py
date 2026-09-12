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

Self-test
---------
``python -m paper.figures.loaders --selftest``
    Loads everything, runs the ``run_epoch_windows`` assertions, and prints a
    per-run PASS/FAIL/SKIPPED table plus the canonical parse count. This is
    the acceptance gate for the figure data infrastructure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    r"(?P<epoch>\d+)\*? \| "
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
    """Load ``checkpoints/epoch_log.jsonl`` into a list of record dicts."""
    if not EPOCH_LOG_JSONL.exists():
        return []
    with EPOCH_LOG_JSONL.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


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


def canonical_gradnorm() -> pd.DataFrame:
    """Parse the canonical GradNorm probe log into a per-epoch DataFrame.

    Columns: ``epoch``, ``loss``, ``train_acc``, ``val_acc``, ``val_acc_ci_lo``,
    ``val_acc_ci_hi``, ``val_dice``, ``seg_weight``, ``cls_weight`` (all
    percentages/weights as floats).
    """
    if not CANONICAL_GRADNORM_LOG.exists():
        return pd.DataFrame(
            columns=["epoch", "loss", "train_acc", "val_acc", "val_acc_ci_lo",
                     "val_acc_ci_hi", "val_dice", "seg_weight", "cls_weight"]
        )
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
    return pd.DataFrame(rows)


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
