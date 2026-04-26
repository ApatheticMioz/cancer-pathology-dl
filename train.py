#!/usr/bin/env python3
"""Lean entrypoint for the Onco 2025 multi-task reproduction pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from repro.config import CHECKPOINT_DIR
from repro.runner import build_arg_parser, run_reproduction


def main() -> int:
    # Make all relative dataset paths resolve from the final/ directory.
    os.chdir(Path(__file__).resolve().parent)

    parser = build_arg_parser()
    args = parser.parse_args()
    args.checkpoint_dir = CHECKPOINT_DIR

    run_reproduction(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
