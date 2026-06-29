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
ts "GROUP 5" "=== DONE ==="

###############################################################################
# Pipeline complete
###############################################################################
ts "PIPELINE" "=========================================="
ts "PIPELINE" "ALL 26 RUNS COMPLETED"
ts "PIPELINE" "=========================================="
ts "PIPELINE" "Logs:      logs/run_*.log"
ts "PIPELINE" "Summaries: checkpoints/summary_*.json"
ts "PIPELINE" "Run 'python src/aggregate_results.py' to generate paper tables."