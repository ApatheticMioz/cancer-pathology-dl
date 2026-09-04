#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAPER_DIR="${ROOT_DIR}/paper"

echo "==> Compiling paper in ${PAPER_DIR}..."
cd "${PAPER_DIR}"

pdflatex -interaction=nonstopmode all_dice_no_slice.tex > /dev/null
bibtex all_dice_no_slice > /dev/null
pdflatex -interaction=nonstopmode all_dice_no_slice.tex > /dev/null
pdflatex -interaction=nonstopmode all_dice_no_slice.tex > /dev/null

if [ ! -f "all_dice_no_slice.pdf" ]; then
    echo "ERROR: all_dice_no_slice.pdf was not generated!" >&2
    exit 1
fi

echo "==> Verifying page count..."
PAGE_COUNT=$(pdfinfo all_dice_no_slice.pdf | awk '/^Pages:/ {print $2}')
echo "    Compiled page count: ${PAGE_COUNT} pages"

if [ "${PAGE_COUNT}" -ne 10 ]; then
    echo "WARNING: Manuscript is ${PAGE_COUNT} pages (target is strictly 10 pages for camera-ready LNCS)!" >&2
else
    echo "    [PASS] Page count strictly adheres to 10-page camera-ready limit."
fi

echo "==> Copying compiled PDF to repository root (${ROOT_DIR}/all_dice_no_slice.pdf)..."
cp all_dice_no_slice.pdf "${ROOT_DIR}/all_dice_no_slice.pdf"

echo "==> Build complete: ${ROOT_DIR}/all_dice_no_slice.pdf"