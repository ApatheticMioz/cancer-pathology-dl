"""Numeric-token drift gate for prose rewrites (invariant I1/I5 companion).

Compares the multiset of numeric tokens (and LaTeX citation keys / labels)
between two versions of a .tex file. Any numeric or citation drift introduced
by a style-only rewrite fails the gate with a nonzero exit.

Usage:
    python scripts/numeric_token_diff.py OLD.tex NEW.tex [--numbers-only]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter

NUMERIC = re.compile(r"\d+(?:\.\d+)?")
CITE_KEY = re.compile(r"\\cite[tp]?\{([^}]+)\}|\\ref\{([^}]+)\}")


def tokens(text: str, numbers_only: bool) -> Counter:
    counts: Counter = Counter(NUMERIC.findall(text))
    if not numbers_only:
        for m in CITE_KEY.finditer(text):
            for key in filter(None, m.groups()):
                for part in key.split(","):
                    counts[f"key:{part.strip()}"] += 1
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--numbers-only", action="store_true",
                   help="Ignore citation/label keys; compare numerics only")
    args = p.parse_args()

    old = tokens(open(args.old, encoding="utf-8").read(), args.numbers_only)
    new = tokens(open(args.new, encoding="utf-8").read(), args.numbers_only)

    lost = old - new
    gained = new - old
    if not lost and not gained:
        print(f"GATE PASS: numeric/citation multiset identical "
              f"({sum(old.values())} tokens)")
        return 0

    print("GATE FAIL: numeric/citation drift detected")
    for tok, n in sorted(lost.items()):
        if isinstance(tok, str) and tok.startswith("key:"):
            print(f"  lost key   {tok[4:]} x{n}")
        else:
            print(f"  lost num   {tok} x{n}")
    for tok, n in sorted(gained.items()):
        if isinstance(tok, str) and tok.startswith("key:"):
            print(f"  gained key {tok[4:]} x{n}")
        else:
            print(f"  gained num {tok} x{n}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
