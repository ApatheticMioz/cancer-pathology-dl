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

# CBM (Elsevier) has no fixed page limit; we only require a non-trivial
# page count (>= 5) to catch degenerate/empty builds.
if [ -z "${PAGE_COUNT}" ] || [ "${PAGE_COUNT}" -lt 5 ]; then
    echo "ERROR: Manuscript compiled to ${PAGE_COUNT:-0} pages (expected a non-trivial page count >= 5)!" >&2
    exit 1
else
    echo "    [PASS] Non-trivial page count (${PAGE_COUNT} pages) for CBM submission."
fi

echo "==> Copying compiled PDF to repository root (${ROOT_DIR}/all_dice_no_slice.pdf)..."
cp all_dice_no_slice.pdf "${ROOT_DIR}/all_dice_no_slice.pdf"

echo "==> Build complete: ${ROOT_DIR}/all_dice_no_slice.pdf"