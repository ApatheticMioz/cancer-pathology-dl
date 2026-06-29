#!/usr/bin/env bash
###############################################################################
# run_remaining.sh
# Re-run script for the 6 experiments that did not complete successfully.
#
# Failed/Incomplete runs:
#   Run 09: g2_panda_vgg16        - FAILED (PANDA macenko data missing)
#   Run 10: g2_panda_mobilenet_v2 - FAILED (PANDA macenko data missing)
#   Run 23: g5_panda_nomacenko    - INTERRUPTED (1 epoch only)
#   Run 24: g5_pannuke_nomacenko  - INTERRUPTED (2 epochs only)
#   Run 25: g5_tcga_noskip        - CRASHED (CUDA illegal memory, epoch 4)
#   Run 26: g5_panda_noskip       - NEVER STARTED (terminated before data load)
#
# Pre-requisite: PANDA Macenko preprocessed data must exist at
#   data/PANDA/preprocessed_macenko/images/
# This script resolves the data dependency in Phase 1.
#
# Hardware target: RTX 3090 (24 GB VRAM) + 12-core CPU, 18 GB system RAM
# Concurrency: exactly 3 parallel Python processes at all times.
# Workers: --num-workers 3 per job to map perfectly to 12-core WSL environment.
#
# Usage:
#   chmod +x run_remaining.sh
#   ./run_remaining.sh
###############################################################################
set -euo pipefail

# ---------------------------------------------------------------------------
# Environment activation
# ---------------------------------------------------------------------------
echo "============================================================"
echo " $(date '+%Y-%m-%d %H:%M:%S') - Activating virtual environment ..."
echo "============================================================"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/venv/bin/activate"
echo " $(date '+%Y-%m-%d %H:%M:%S') - Virtual environment active: $(which python)"

# ---------------------------------------------------------------------------
# Concurrency control (max 3 parallel Python processes)
# ---------------------------------------------------------------------------
MAX_JOBS=3
declare -a PIDS=()

launch_job() {
    local run_id="$1"
    local run_name="$2"
    shift 2
    local cmd=("$@")

    local log_file="logs/run_$(printf '%02d' ${run_id})_${run_name}.log"
    local summary_file="checkpoints/summary_$(printf '%02d' ${run_id})_${run_name}.json"

    mkdir -p logs checkpoints

    echo " [${run_id}] $(date '+%Y-%m-%d %H:%M:%S') - START: ${run_name}"
    echo "        Log:      ${log_file}"
    echo "        Summary: ${summary_file}"

    "${cmd[@]}" --summary-out "${summary_file}" >> "${log_file}" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    echo " [${run_id}] PID: ${pid}"
}

# Reap finished PIDs and return count of still-running jobs
count_active() {
    local active=0
    local -a surviving=()
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            active=$((active + 1))
            surviving+=("$pid")
        fi
    done
    PIDS=("${surviving[@]}")
    echo "$active"
}

wait_for_slot() {
    while true; do
        local active
        active=$(count_active)
        if [ "$active" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 2
    done
}

# Wait for ALL remaining background jobs and clear PID array
wait_all() {
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    PIDS=()
}

# ---------------------------------------------------------------------------
# Helper: timestamped echo
# ---------------------------------------------------------------------------
ts() {
    echo " [$1] $(date '+%Y-%m-%d %H:%M:%S') - $2"
}

# ---------------------------------------------------------------------------
# Ensure output directories
# ---------------------------------------------------------------------------
mkdir -p logs checkpoints

# ===========================================================================
# PHASE 1: Resolve PANDA Macenko preprocessed data dependency
# ===========================================================================
# Runs 9 and 10 require Macenko-normalized PANDA images at:
#   data/PANDA/preprocessed_macenko/images/
#
# apply_macenko.py outputs to:
#   data/PANDA/preprocessed_macenko_fixed/images/
#
# Strategy:
#   1. If preprocessed_macenko/images/ already has files -> skip
#   2. If preprocessed_macenko_fixed/images/ has files -> symlink it
#   3. Otherwise -> run apply_macenko.py to generate, then symlink
# ===========================================================================
ts "PHASE1" "=========================================="
ts "PHASE1" "Resolving PANDA Macenko data dependency ..."
ts "PHASE1" "=========================================="

PANDA_MACENKO_DIR="data/PANDA/preprocessed_macenko/images"
PANDA_FIXED_DIR="data/PANDA/preprocessed_macenko_fixed/images"
PANDA_LINK_DIR="data/PANDA/preprocessed_macenko"

# Use absolute paths for symlink target (symlinks resolve relative to their parent dir)
PANDA_FIXED_ABS="$(cd "${SCRIPT_DIR}" && realpath "${PANDA_FIXED_DIR}")"

if [ -d "${PANDA_MACENKO_DIR}" ] && [ "$(ls -A "${PANDA_MACENKO_DIR}" 2>/dev/null | head -1)" ]; then
    ts "PHASE1" "PANDA Macenko data already exists at ${PANDA_MACENKO_DIR}"
    ts "PHASE1" "Skipping Macenko setup."
elif [ -d "${PANDA_FIXED_DIR}" ] && [ "$(ls -A "${PANDA_FIXED_DIR}" 2>/dev/null | head -1)" ]; then
    ts "PHASE1" "Found Macenko data at ${PANDA_FIXED_DIR} (from previous apply_macenko.py run)"
    mkdir -p "${PANDA_LINK_DIR}"
    if [ ! -e "${PANDA_LINK_DIR}/images" ]; then
        ln -sf "${PANDA_FIXED_ABS}" "${PANDA_LINK_DIR}/images"
        ts "PHASE1" "Created symlink: ${PANDA_LINK_DIR}/images -> ${PANDA_FIXED_ABS}"
    else
        ts "PHASE1" "Symlink ${PANDA_LINK_DIR}/images already exists."
    fi
    ts "PHASE1" "PANDA Macenko data ready."
else
    ts "PHASE1" "PANDA Macenko data MISSING in both locations."
    ts "PHASE1" "Running apply_macenko.py to generate PANDA Macenko data ..."
    ts "PHASE1" "This may take 30-60 minutes depending on CPU cores."

    python src/apply_macenko.py --datasets panda --workers 12

    if [ -d "${PANDA_FIXED_DIR}" ] && [ "$(ls -A "${PANDA_FIXED_DIR}" 2>/dev/null | head -1)" ]; then
        mkdir -p "${PANDA_LINK_DIR}"
        if [ ! -e "${PANDA_LINK_DIR}/images" ]; then
            ln -sf "${PANDA_FIXED_ABS}" "${PANDA_LINK_DIR}/images"
            ts "PHASE1" "Created symlink: ${PANDA_LINK_DIR}/images -> ${PANDA_FIXED_ABS}"
        fi
        ts "PHASE1" "PANDA Macenko data ready."
    else
        ts "PHASE1" "ERROR: apply_macenko.py did not produce output at ${PANDA_FIXED_DIR}"
        ts "PHASE1" "Aborting. Fix the Macenko data and re-run."
        exit 1
    fi
fi

# ===========================================================================
# PHASE 2: Re-run the 6 failed/incomplete experiments
# ===========================================================================
ts "PHASE2" "=========================================="
ts "PHASE2" "REMAINING RUNS - 6 experiments"
ts "PHASE2" "=========================================="

# ---------------------------------------------------------------------------
# Runs 9-10: Group 2 PANDA runs (phase v2, Macenko ON, GradNorm ON)
# These failed because PANDA preprocessed_macenko/images was missing.
# Phase 1 has resolved the data dependency.
# Note: --phase v2 enables GradNorm, which auto-disables torch.compile
#       (see main.py:487-492), so no --compile flag needed.
# ---------------------------------------------------------------------------
ts "PHASE2" "--- Group 2 PANDA runs (data dependency resolved) ---"

wait_for_slot; launch_job 9  "g2_panda_vgg16"        python main.py --phase v2 --datasets panda  --encoders vgg16        --num-workers 3
wait_for_slot; launch_job 10 "g2_panda_mobilenet_v2" python main.py --phase v2 --datasets panda  --encoders mobilenet_v2 --num-workers 3

ts "PHASE2" "Runs 9-10 launched. Waiting for completion ..."
wait_all
ts "PHASE2" "Runs 9-10 complete."

# ---------------------------------------------------------------------------
# Runs 23-26: Group 5 ablation runs (phase v2)
# Run 23: INTERRUPTED after 1 epoch
# Run 24: INTERRUPTED after 2 epochs
# Run 25: CRASHED with CUDA illegal memory at epoch 4
#         -> Added --no-compile to prevent torch.compile from generating
#            bad CUDA kernels that caused cudaErrorIllegalAddress
# Run 26: NEVER STARTED (terminated before data load)
# All will start fresh (--no-resume is default behavior of main.py)
# ---------------------------------------------------------------------------
ts "PHASE2" "--- Group 5 ablation runs (fresh start) ---"

wait_for_slot; launch_job 23 "g5_panda_nomacenko"      python main.py --phase v2 --datasets panda  --encoders mobilenet_v2 --no-macenko --num-workers 3
wait_for_slot; launch_job 24 "g5_pannuke_nomacenko"    python main.py --phase v2 --datasets pannuke --encoders mobilenet_v2 --no-macenko --num-workers 3
wait_for_slot; launch_job 25 "g5_tcga_noskip"          python main.py --phase v2 --datasets tcga   --encoders mobilenet_v2 --no-skip-connections --no-compile --num-workers 3
wait_for_slot; launch_job 26 "g5_panda_noskip"         python main.py --phase v2 --datasets panda  --encoders mobilenet_v2 --no-skip-connections --num-workers 3

ts "PHASE2" "Runs 23-26 launched. Waiting for completion ..."
wait_all
ts "PHASE2" "Runs 23-26 complete."

# ===========================================================================
# Pipeline complete
# ===========================================================================
ts "PIPELINE" "=========================================="
ts "PIPELINE" "ALL 6 REMAINING RUNS COMPLETED"
ts "PIPELINE" "=========================================="
ts "PIPELINE" "Logs:      logs/run_09_*.log  logs/run_10_*.log  logs/run_2{3,4,5,6}_*.log"
ts "PIPELINE" "Summaries: checkpoints/summary_09_*.json  ...  summary_26_*.json"
ts "PIPELINE" "Run 'python src/aggregate_results.py' to generate paper tables."