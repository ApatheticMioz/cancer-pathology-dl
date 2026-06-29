#!/usr/bin/env bash
###############################################################################
# run_all_experiments.sh
# Master orchestrator for the definitive 26-run experimental matrix.
#
# Hardware target: RTX 3090 (24 GB VRAM) + 12-core CPU, 18 GB system RAM
# Concurrency: exactly 3 parallel Python processes at all times.
# Workers: --num-workers 3 per job to map perfectly to 12-core WSL environment.
#
# Collision safety:
#   - Each run writes to a unique log file:    logs/run_XX_[Name].log
#   - Each run writes to a unique summary:    checkpoints/summary_XX_[Name].json
#   - --no-resume enforced to guarantee clean, independent runs
#
# Usage:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh
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

    run_with_safe_healing "$run_id" "$run_name" "$log_file" \
        "${cmd[@]}" --summary-out "${summary_file}" >> /dev/null 2>&1 &
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

# Scan log files for a given group range and report any permanent failures
check_group_failures() {
    local group_label="$1"
    local start_id="$2"
    local end_id="$3"
    local failures=0

    for id in $(seq "$start_id" "$end_id"); do
        local log_file
        log_file=$(ls "logs/run_$(printf '%02d' ${id})_*.log" 2>/dev/null | head -1)
        if [ -n "$log_file" ] && grep -q "PERMANENT FAILURE" "$log_file"; then
            echo "  ${RED}[${id}] PERMANENT FAILURE detected: $log_file${NC}"
            failures=$((failures + 1))
        fi
    done

    if [ "$failures" -gt 0 ]; then
        echo -e "  ${RED}WARNING: $failures run(s) in ${group_label} failed permanently.${NC}"
        echo -e "  ${RED}Continuing with remaining groups. Review logs/ after pipeline completes.${NC}"
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Helper: timestamped echo
# ---------------------------------------------------------------------------
ts() {
    echo " [$1] $(date '+%Y-%m-%d %H:%M:%S') - $2"
}

# ---------------------------------------------------------------------------
# Safe Auto-Healing Wrapper
# ---------------------------------------------------------------------------
# Retries a failed run with reduced system-level optimizations to handle
# transient OOM, torch.compile crashes, or DataLoader worker spikes.
# Does NOT alter any hyperparameters (batch size, LR, epochs, loss weights).
#
# Sequence:
#   Attempt 1 — Run command as-is.
#   Attempt 2 — Sleep 10s for VRAM flush, then retry with:
#     REPRO_TORCH_COMPILE_BACKEND="none"     (disables torch.compile)
#     REPRO_ALLOW_BIG_CACHE="false"          (limits RAM cache)
#     REPRO_ALLOW_UNC_WORKERS="0"            (disables DataLoader workers)
#   Failure  — Log permanent error; caller decides whether to abort.
# ---------------------------------------------------------------------------
run_with_safe_healing() {
    local run_id="$1"
    local run_name="$2"
    local log_file="$3"
    shift 3
    local cmd=("$@")

    # ── Attempt 1: Standard run ──────────────────────────────────────────
    {
        echo "[$run_id] $(date '+%Y-%m-%d %H:%M:%S') - [ATTEMPT 1/2] Starting standard run..."
        echo "[$run_id] Command: ${cmd[*]}"
    } >> "$log_file" 2>&1

    "${cmd[@]}" >> "$log_file" 2>&1
    local rc=$?

    if [ $rc -eq 0 ]; then
        {
            echo "[$run_id] $(date '+%Y-%m-%d %H:%M:%S') - [ATTEMPT 1/2] SUCCESS"
        } >> "$log_file" 2>&1
        return 0
    fi

    {
        echo "[$run_id] $(date '+%Y-%m-%d %H:%M:%S') - [ATTEMPT 1/2] FAILED (exit code $rc)"
        echo "[$run_id] Last 30 lines of Attempt 1:"
        tail -n 30 "$log_file"
        echo "[$run_id] Sleeping 10s for GPU VRAM flush before Attempt 2..."
    } >> "$log_file" 2>&1

    sleep 10

    # ── Attempt 2: Safe Memory/Compile Fallback ─────────────────────────
    {
        echo "[$run_id] $(date '+%Y-%m-%d %H:%M:%S') - [ATTEMPT 2/2] Starting safe fallback run..."
        echo "[$run_id] Env: REPRO_TORCH_COMPILE_BACKEND=none REPRO_ALLOW_BIG_CACHE=false REPRO_ALLOW_UNC_WORKERS=0"
        echo "[$run_id] Command: ${cmd[*]}"
    } >> "$log_file" 2>&1

    REPRO_TORCH_COMPILE_BACKEND="none" \
    REPRO_ALLOW_BIG_CACHE="false" \
    REPRO_ALLOW_UNC_WORKERS="0" \
    "${cmd[@]}" >> "$log_file" 2>&1
    rc=$?

    if [ $rc -eq 0 ]; then
        {
            echo "[$run_id] $(date '+%Y-%m-%d %H:%M:%S') - [ATTEMPT 2/2] SUCCESS (recovered from Attempt 1 failure)"
        } >> "$log_file" 2>&1
        return 0
    fi

    # ── Permanent Failure ────────────────────────────────────────────────
    {
        echo "[$run_id] $(date '+%Y-%m-%d %H:%M:%S') - [ATTEMPT 2/2] FAILED (exit code $rc)"
        echo "[$run_id] *** PERMANENT FAILURE for ${run_name} ***"
        echo "[$run_id] Both attempts exhausted. Check $log_file for details."
        echo "[$run_id] Last 30 lines of Attempt 2:"
        tail -n 30 "$log_file"
    } >> "$log_file" 2>&1

    return $rc
}

# ---------------------------------------------------------------------------
# Ensure output directories
# ---------------------------------------------------------------------------
mkdir -p logs checkpoints

###############################################################################
# GROUP 1: The Naked Baseline (6 Runs)
# TCGA, PANDA, SIIM x VGG16, MobileNetV2
# --phase v1, --no-macenko, --disable-gradnorm, --static-weights
# (Original paper LR 1e-3, 5:1 static weights)
###############################################################################
ts "GROUP 1" "=== The Naked Baseline (6 runs) ==="

wait_for_slot; launch_job 1  "g1_tcga_vgg16"        python main.py --phase v1 --datasets tcga   --encoders vgg16        --no-macenko --disable-gradnorm --static-weights --compile --num-workers 3
wait_for_slot; launch_job 2  "g1_tcga_mobilenet_v2" python main.py --phase v1 --datasets tcga   --encoders mobilenet_v2 --no-macenko --disable-gradnorm --static-weights --num-workers 3
wait_for_slot; launch_job 3  "g1_panda_vgg16"       python main.py --phase v1 --datasets panda  --encoders vgg16        --no-macenko --disable-gradnorm --static-weights --compile --num-workers 3
wait_for_slot; launch_job 4  "g1_panda_mobilenet_v2" python main.py --phase v1 --datasets panda  --encoders mobilenet_v2 --no-macenko --disable-gradnorm --static-weights --compile --num-workers 3
wait_for_slot; launch_job 5  "g1_siim_vgg16"        python main.py --phase v1 --datasets siim   --encoders vgg16        --no-macenko --disable-gradnorm --static-weights --compile --num-workers 3
wait_for_slot; launch_job 6  "g1_siim_mobilenet_v2" python main.py --phase v1 --datasets siim   --encoders mobilenet_v2 --no-macenko --disable-gradnorm --static-weights --compile --num-workers 3

ts "GROUP 1" "All 6 jobs launched. Waiting for completion ..."
wait_all
check_group_failures "GROUP 1" 1 6
ts "GROUP 1" "=== DONE ==="

###############################################################################
# GROUP 2: The Final Form Package (6 Runs)
# TCGA, PANDA, SIIM x VGG16, MobileNetV2
# --phase v2 (GradNorm ON, New LR 1e-4, Macenko ON by default)
###############################################################################
ts "GROUP 2" "=== The Final Form Package (6 runs) ==="

wait_for_slot; launch_job 7  "g2_tcga_vgg16"        python main.py --phase v2 --datasets tcga   --encoders vgg16        --num-workers 3
wait_for_slot; launch_job 8  "g2_tcga_mobilenet_v2" python main.py --phase v2 --datasets tcga   --encoders mobilenet_v2 --num-workers 3
wait_for_slot; launch_job 9  "g2_panda_vgg16"       python main.py --phase v2 --datasets panda  --encoders vgg16        --num-workers 3
wait_for_slot; launch_job 10 "g2_panda_mobilenet_v2" python main.py --phase v2 --datasets panda  --encoders mobilenet_v2 --num-workers 3
wait_for_slot; launch_job 11 "g2_siim_vgg16"        python main.py --phase v2 --datasets siim   --encoders vgg16        --num-workers 3
wait_for_slot; launch_job 12 "g2_siim_mobilenet_v2" python main.py --phase v2 --datasets siim   --encoders mobilenet_v2 --num-workers 3

ts "GROUP 2" "All 6 jobs launched. Waiting for completion ..."
wait_all
check_group_failures "GROUP 2" 7 12
ts "GROUP 2" "=== DONE ==="

###############################################################################
# GROUP 3: The PanNuke Crucible (4 Runs)
# PanNuke x VGG16, MobileNetV2
#   A (Naked): --phase v1, --no-macenko, --disable-gradnorm, --static-weights
#   B (Final): --phase v2 (GradNorm ON, New LR, Macenko ON)
###############################################################################
ts "GROUP 3" "=== The PanNuke Crucible (4 runs) ==="

wait_for_slot; launch_job 13 "g3_pannuke_vgg16_naked"      python main.py --phase v1 --datasets pannuke --encoders vgg16        --no-macenko --disable-gradnorm --static-weights --compile --num-workers 3
wait_for_slot; launch_job 14 "g3_pannuke_mobilenet_v2_naked" python main.py --phase v1 --datasets pannuke --encoders mobilenet_v2 --no-macenko --disable-gradnorm --static-weights --compile --num-workers 3
wait_for_slot; launch_job 15 "g3_pannuke_vgg16_final"      python main.py --phase v2 --datasets pannuke --encoders vgg16        --num-workers 3
wait_for_slot; launch_job 16 "g3_pannuke_mobilenet_v2_final" python main.py --phase v2 --datasets pannuke --encoders mobilenet_v2 --num-workers 3

ts "GROUP 3" "All 4 jobs launched. Waiting for completion ..."
wait_all
check_group_failures "GROUP 3" 13 16
ts "GROUP 3" "=== DONE ==="

###############################################################################
# GROUP 4: The Optimization Teardown (6 Runs - PANDA x VGG16 Only)
#
#   4.1 Isolate LR:     --phase v2, --no-macenko, --disable-gradnorm, --static-weights
#   4.2 Isolate GradNorm: --phase v1, --no-macenko (GradNorm ON implicitly)
#   4.3-4.6 Lambda Sweeps: --phase v1, --no-macenko, --disable-gradnorm, --static-weights
#       [1:1], [5:1], [1:10], [10:1]
###############################################################################
ts "GROUP 4" "=== The Optimization Teardown (6 runs) ==="

wait_for_slot; launch_job 17 "g4_panda_isolate_lr"     python main.py --phase v2 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --num-workers 3
wait_for_slot; launch_job 18 "g4_panda_isolate_gn"     python main.py --phase v1 --datasets panda --encoders vgg16 --no-macenko --num-workers 3
wait_for_slot; launch_job 19 "g4_panda_lambda_1_1"     python main.py --phase v1 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --lambda-seg 1  --lambda-cls 1  --compile --num-workers 3
wait_for_slot; launch_job 20 "g4_panda_lambda_5_1"     python main.py --phase v1 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --lambda-seg 5  --lambda-cls 1  --compile --num-workers 3
wait_for_slot; launch_job 21 "g4_panda_lambda_1_10"    python main.py --phase v1 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --lambda-seg 1  --lambda-cls 10 --compile --num-workers 3
wait_for_slot; launch_job 22 "g4_panda_lambda_10_1"    python main.py --phase v1 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --lambda-seg 10 --lambda-cls 1  --compile --num-workers 3

ts "GROUP 4" "All 6 jobs launched. Waiting for completion ..."
wait_all
check_group_failures "GROUP 4" 17 22
ts "GROUP 4" "=== DONE ==="

###############################################################################
# GROUP 5: Preprocessing & Architecture Ablations (4 Runs)
#
#   5.1-5.2 Macenko Truth: PANDA & PanNuke x MobileNetV2, --phase v2, --no-macenko
#   5.3-5.4 Architecture:  TCGA & PANDA x MobileNetV2, --phase v2, --no-skip-connections
###############################################################################
ts "GROUP 5" "=== Preprocessing & Architecture Ablations (4 runs) ==="

wait_for_slot; launch_job 23 "g5_panda_nomacenko"      python main.py --phase v2 --datasets panda  --encoders mobilenet_v2 --no-macenko --num-workers 3
wait_for_slot; launch_job 24 "g5_pannuke_nomacenko"    python main.py --phase v2 --datasets pannuke --encoders mobilenet_v2 --no-macenko --num-workers 3
wait_for_slot; launch_job 25 "g5_tcga_noskip"          python main.py --phase v2 --datasets tcga   --encoders mobilenet_v2 --no-skip-connections --num-workers 3
wait_for_slot; launch_job 26 "g5_panda_noskip"         python main.py --phase v2 --datasets panda  --encoders mobilenet_v2 --no-skip-connections --num-workers 3

ts "GROUP 5" "All 4 jobs launched. Waiting for completion ..."
wait_all
check_group_failures "GROUP 5" 23 26
ts "GROUP 5" "=== DONE ==="

###############################################################################
# Pipeline complete
###############################################################################
ts "PIPELINE" "=========================================="
ts "PIPELINE" "ALL 26 RUNS COMPLETED"
ts "PIPELINE" "=========================================="
ts "PIPELINE" "Logs:      logs/run_*.log"
ts "PIPELINE" "Summaries: checkpoints/summary_*.json"

# ---------------------------------------------------------------------------
# Aggregate results into paper-ready CSV and LaTeX tables
# ---------------------------------------------------------------------------
ts "PIPELINE" "Running results aggregator ..."
if python src/aggregate_results.py; then
    ts "PIPELINE" "Aggregation successful."
    ts "PIPELINE" "CSV:   paper/paper_results_matrix.csv"
    ts "PIPELINE" "LaTeX: paper/paper_results_latex_table.txt"
else
    ts "PIPELINE" "Aggregation failed (some runs may not have completed)."
    ts "PIPELINE" "Re-run failed jobs manually, then execute 'python src/aggregate_results.py'."
fi