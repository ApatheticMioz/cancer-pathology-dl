#!/usr/bin/env python3
"""Report-only consistency gate: paper/all_dice_no_slice.tex vs the results CSV.

Extracts every numeric literal from the manuscript ``.tex`` and checks whether
each one is present (within a small rounding tolerance) among the numeric
values carried by the results CSV (``paper/paper_results_matrix_with_ci.csv``
by default).  Any tex number that is NOT in the CSV and NOT covered by the
allowlist is flagged as a potential inconsistency.

Design constraints (HARD):
  * REPORT-ONLY: this script NEVER writes to or edits the ``.tex``.  It only
    reads the tex and the CSV and prints a deterministic report.
  * Tolerant matching: a tex value matches a CSV value if they are equal after
    rounding to 2 decimal places, or if they differ by at most ``--tol``
    (default 0.005).  This absorbs ``±SD`` suffixes (the mean and the SD are
    extracted as separate tokens) and percent formatting (``88\\%`` -> 88.0).
  * Allowlist mechanism: years (1900-2099) are allowed by default; section
    numbers / run ids / fold numbers (small ints) and 4-digit citation years can
    be allowed via flags; an explicit ``--allowlist FILE`` (one value per line)
    is honoured exactly.

Exit code: 0 when no unexplained numbers remain (gate passes), 1 when at least
one tex number is neither in the CSV nor allowlisted (gate fails).  The tex is
never modified in either case.

Usage:
    python3 scripts/check_tex_csv_consistency.py \
        [--tex paper/all_dice_no_slice.tex] \
        [--csv paper/paper_results_matrix_with_ci.csv] \
        [--allowlist FILE] [--allow-years] [--allow-small-ints N] \
        [--allow-4digit] [--tol 0.005]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEX = PROJECT_ROOT / "paper" / "all_dice_no_slice.tex"
DEFAULT_CSV = PROJECT_ROOT / "paper" / "paper_results_matrix_with_ci.csv"

# Numeric literal: an integer or decimal, optionally with a thousands
# separator (``1{,}659``) or a LaTeX exponent (``10^{-3}``).  We capture the
# base mantissa and handle the exponent / separator in _parse_token.
_NUM_RE = re.compile(r"(?<![\w.])\d{1,3}(?:\{,\}\d{3})+|\d+(?:\.\d+)?")


def _parse_token(tok: str) -> float | None:
    """Convert a raw numeric token to a float, or None if not a plain number.

    Handles thousands separators (``1{,}659`` -> 1659) and LaTeX exponents
    (``10^{-3}`` -> 0.001).  Returns None for tokens that are not standalone
    numbers (e.g. a bare ``10`` that is actually part of ``10^{-3}`` is handled
    by the caller which strips the exponent first).
    """
    t = tok
    # Thousands separator: 1{,}659 -> 1659
    if "{,}" in t:
        t = t.replace("{,}", "")
    # LaTeX exponent: 10^{-3} means 10 raised to -3 = 0.001 (a single number).
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\^\{(-?\d+)\}", t)
    if m:
        return float(m.group(1)) ** int(m.group(2))
    try:
        return float(t)
    except ValueError:
        return None


def extract_tex_numbers(tex_path: Path) -> list[tuple[int, str, float]]:
    """Return [(line_no, raw_token, value)] for every numeric literal in the tex.

    Exponent tokens (``10^{-3}``) are matched as a whole so the mantissa is not
    double-counted.  Comment lines (starting with ``%``) are skipped.
    """
    out: list[tuple[int, str, float]] = []
    # Match exponent form first (longest), then plain numbers.
    exp_re = re.compile(r"\d+(?:\.\d+)?\^\{-?\d+\}")
    plain_re = re.compile(r"(?<![\w.])\d{1,3}(?:\{,\}\d{3})+|(?<![\w.])\d+(?:\.\d+)?")
    for line_no, line in enumerate(tex_path.read_text().splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        # Mask exponent tokens so their mantissa is not re-matched as a plain
        # number, then record the exponent value itself.
        masked = line
        for m in exp_re.finditer(line):
            val = _parse_token(m.group(0))
            if val is not None:
                out.append((line_no, m.group(0), val))
            masked = masked[:m.start()] + (" " * (m.end() - m.start())) + masked[m.end():]
        for m in plain_re.finditer(masked):
            val = _parse_token(m.group(0))
            if val is not None:
                out.append((line_no, m.group(0), val))
    return out


def extract_csv_values(csv_path: Path) -> set[float]:
    """Collect every numeric value from every cell of the CSV (2dp-rounded)."""
    vals: set[float] = set()
    num_re = re.compile(r"-?\d+(?:\.\d+)?")
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            for cell in row.values():
                if cell is None:
                    continue
                for m in num_re.finditer(str(cell)):
                    try:
                        vals.add(round(float(m.group(0)), 2))
                    except ValueError:
                        pass
    return vals


def load_allowlist(path: Path) -> set[float]:
    """Load an explicit allowlist file: one value per line (``#`` comments ok)."""
    allowed: set[float] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            allowed.add(round(float(line), 2))
        except ValueError:
            print(f"  [WARN] allowlist: skipping non-numeric line {line!r}", file=sys.stderr)
    return allowed


def _is_allowed(
    val: float,
    allowlist: set[float],
    allow_years: bool,
    allow_small_ints: int,
    allow_4digit: bool,
) -> bool:
    if round(val, 2) in allowlist:
        return True
    if allow_years and 1900 <= val <= 2099:
        return True
    if allow_small_ints and val == int(val) and 1 <= int(val) <= allow_small_ints:
        return True
    if allow_4digit and val == int(val) and 1000 <= int(val) <= 9999:
        return True
    return False


def _matches_csv(val: float, csv_vals: set[float], tol: float) -> bool:
    r = round(val, 2)
    if r in csv_vals:
        return True
    for v in csv_vals:
        if abs(v - val) <= tol:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--allowlist", type=Path, default=None,
                        help="file with one allowed value per line")
    parser.add_argument("--allow-years", action="store_true", default=True,
                        help="allow 1900-2099 (default on)")
    parser.add_argument("--no-allow-years", dest="allow_years", action="store_false")
    parser.add_argument("--allow-small-ints", type=int, default=0, metavar="N",
                        help="allow integers 1..N (section/run/fold numbers)")
    parser.add_argument("--allow-4digit", action="store_true",
                        help="allow 4-digit integers (citation years)")
    parser.add_argument("--tol", type=float, default=0.005,
                        help="rounding tolerance for CSV matching (default 0.005)")
    args = parser.parse_args()

    if not args.tex.is_file():
        print(f"[ERROR] tex not found: {args.tex}", file=sys.stderr)
        return 2
    if not args.csv.is_file():
        print(f"[ERROR] csv not found: {args.csv}", file=sys.stderr)
        return 2

    allowlist = load_allowlist(args.allowlist) if args.allowlist else set()
    csv_vals = extract_csv_values(args.csv)
    tex_nums = extract_tex_numbers(args.tex)

    flagged: list[tuple[int, str, float]] = []
    matched = 0
    allowed = 0
    for line_no, tok, val in tex_nums:
        if _matches_csv(val, csv_vals, args.tol):
            matched += 1
        elif _is_allowed(val, allowlist, args.allow_years, args.allow_small_ints,
                         args.allow_4digit):
            allowed += 1
        else:
            flagged.append((line_no, tok, val))

    # Deterministic report: sort by (line, value, token).
    flagged.sort(key=lambda t: (t[0], t[2], t[1]))

    print("=" * 72)
    print("  tex <-> CSV numeric consistency gate (REPORT-ONLY; tex never edited)")
    print("=" * 72)
    print(f"  tex:  {args.tex}")
    print(f"  csv:  {args.csv}")
    print(f"  csv distinct values: {len(csv_vals)}")
    print(f"  tex numeric literals: {len(tex_nums)}")
    print(f"  matched in CSV:       {matched}")
    print(f"  allowed (allowlist):  {allowed}")
    print(f"  FLAGGED (not in CSV, not allowed): {len(flagged)}")
    print("-" * 72)
    if flagged:
        print(f"  {'line':>5}  {'token':>12}  {'value':>10}")
        for line_no, tok, val in flagged:
            print(f"  {line_no:>5}  {tok:>12}  {val:>10.4g}")
        print("-" * 72)
        print("  GATE: FAIL -- unexplained numeric literals present in tex.")
        print("  (These may be legitimate prose/hyperparameters; add them to the")
        print("   allowlist or the corresponding CSV column if they are metrics.)")
        return 1
    print("  GATE: PASS -- every tex number is in the CSV or allowlisted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
