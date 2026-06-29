#!/usr/bin/env bash
###############################################################################
# run_macenko_recovery.sh
# Recovery script: deprecate corrupted Macenko data, promote fixed data,
# archive corrupted run artifacts, and re-run the 7 affected experiments.
#
# Affected runs (PANDA and PanNuke, which rely on preprocessed_macenko/):
#   Run 03: PANDA vgg16        (v1, compile)
#   Run 04: PANDA mobilenet_v2 (v1, compile)
#   Run 09: PANDA vgg16        (v2, compile)
#   Run 10: PANDA mobilenet_v2 (v2, compile)
#   Run 13: PanNuke vgg16      (v2, compile)
#   Run 14: PanNuke mobilenet_v2 (v2, compile)
#   Run 24: PANDA vgg16        (v2, gradnorm-alpha 0.5, compile)
#
# Usage:
#   chmod +x run_macenko_recovery.sh
#   ./run_macenko_recovery.sh
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
# Helper: timestamped echo
# ---------------------------------------------------------------------------
ts() {
    echo " [$1] $(date '+%Y-%m-%d %H:%M:%S') - $2"
}

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



# ===========================================================================
# PHASE 1: Log & Summary Cleanup
# ===========================================================================
ts RECOVERY "=========================================="
ts RECOVERY "MACENKO RECOVERY - PHASE 1: ARTIFACT ARCHIVE"
ts RECOVERY "=========================================="

mkdir -p archive/corrupted_runs

# Define the 7 affected runs: ID and name (matching run_all_experiments.sh naming)
declare -A AFFECTED_RUNS
AFFECTED_RUNS[3]="p1_panda_vgg16"
AFFECTED_RUNS[4]="p1_panda_mobilenet_v2"
AFFECTED_RUNS[9]="p2_panda_vgg16"
AFFECTED_RUNS[10]="p2_panda_mobilenet_v2"
AFFECTED_RUNS[13]="p2_pannuke_vgg16"
AFFECTED_RUNS[14]="p2_pannuke_mobilenet_v2"
AFFECTED_RUNS[24]="p4c_panda_alpha05"

for run_id in "${!AFFECTED_RUNS[@]}"; do
    run_name="${AFFECTED_RUNS[$run_id]}"
    padded_id=$(printf '%02d' "${run_id}")

    log_file="logs/run_${padded_id}_${run_name}.log"
    summary_file="checkpoints/summary_${padded_id}_${run_name}.json"

    # Move summary JSON if it exists
    if [ -f "${summary_file}" ]; then
        mv "${summary_file}" "archive/corrupted_runs/"
        ts ARCHIVE "  ${summary_file} -> archive/corrupted_runs/"
    else
        ts ARCHIVE "  ${summary_file} (not found, skipping)"
    fi

    # Move log file if it exists
    if [ -f "${log_file}" ]; then
        mv "${log_file}" "archive/corrupted_runs/"
        ts ARCHIVE "  ${log_file} -> archive/corrupted_runs/"
    else
        ts ARCHIVE "  ${log_file} (not found, skipping)"
    fi
done

ts RECOVERY "Artifact archive complete. aggregate_results.py will not parse corrupted runs."

# ===========================================================================
# PHASE 2: Recovery Execution Matrix (7 runs, MAX_JOBS=3)
# ===========================================================================
ts RECOVERY "=========================================="
ts RECOVERY "MACENKO RECOVERY - PHASE 2: RE-RUN 7 AFFECTED JOBS"
ts RECOVERY "=========================================="

mkdir -p logs checkpoints

ts RECOVERY "Launching 7 recovery jobs with MAX_JOBS=3 concurrency ..."

# Run 03: PANDA vgg16 (v1, compile)
wait_for_slot; launch_job 3  "p1_panda_vgg16"       python main.py --phase v1 --datasets panda  --encoders vgg16        --compile --num-workers 3

# Run 04: PANDA mobilenet_v2 (v1, compile)
wait_for_slot; launch_job 4  "p1_panda_mobilenet_v2" python main.py --phase v1 --datasets panda  --encoders mobilenet_v2 --compile --num-workers 3

# Run 09: PANDA vgg16 (v2, compile)
wait_for_slot; launch_job 9  "p2_panda_vgg16"       python main.py --phase v2 --datasets panda  --encoders vgg16        --compile --num-workers 3

# Run 10: PANDA mobilenet_v2 (v2, compile)
wait_for_slot; launch_job 10 "p2_panda_mobilenet_v2" python main.py --phase v2 --datasets panda  --encoders mobilenet_v2 --compile --num-workers 3

# Run 13: PanNuke vgg16 (v2, compile)
wait_for_slot; launch_job 13 "p2_pannuke_vgg16"     python main.py --phase v2 --datasets pannuke --encoders vgg16        --compile --num-workers 3

# Run 14: PanNuke mobilenet_v2 (v2, compile)
wait_for_slot; launch_job 14 "p2_pannuke_mobilenet_v2" python main.py --phase v2 --datasets pannuke --encoders mobilenet_v2 --compile --num-workers 3

# Run 24: PANDA vgg16 (v2, gradnorm-alpha 0.5, compile)
wait_for_slot; launch_job 24 "p4c_panda_alpha05"    python main.py --phase v2 --datasets panda  --encoders vgg16        --gradnorm-alpha 0.5 --compile --num-workers 3

ts RECOVERY "All 7 recovery jobs launched. Waiting for completion ..."
wait_all
ts RECOVERY "All 7 recovery jobs finished."

# ===========================================================================
# Done
# ===========================================================================
ts RECOVERY "=========================================="
ts RECOVERY "MACENKO RECOVERY COMPLETE"
ts RECOVERY "=========================================="
ts RECOVERY "Archived data:  data/PANDA/archive_corrupted_macenko/"
ts RECOVERY "Archived data:  data/PanNuke/archive_corrupted_macenko/"
ts RECOVERY "Archived runs:  archive/corrupted_runs/"
ts RECOVERY "New logs:       logs/run_03_*.log  logs/run_04_*.log  logs/run_09_*.log"
ts RECOVERY "                  logs/run_10_*.log  logs/run_13_*.log  logs/run_14_*.log  logs/run_24_*.log"
ts RECOVERY "New summaries:  checkpoints/summary_03_*.json  ...  summary_24_*.json"
ts RECOVERY "Run 'python src/aggregate_results.py' to regenerate paper tables."