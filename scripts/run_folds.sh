#!/usr/bin/env bash
###############################################################################
# run_folds.sh
# 5-fold cross-validation campaign driver.
#
# Re-runs a SELECTED subset of the definitive 26-run matrix (defined in
# run_all_experiments.sh / src/aggregate_results.py EXPECTED_RUNS) under
# group-aware 5-fold CV. Each selected run executes ALL 5 folds IN ONE Python
# process (main.py --k-folds 5 -> src/training.py::train_kfold_cv), which
# aggregates mean +/- std across folds and writes a single consolidated
# summary JSON. There is NO per-fold process loop here: one process per run.
#
# Per-run outputs (collision-safe, mirroring run_all_experiments.sh):
#   - Log:      logs/kfold_<label>.log
#   - Summary:  results/round2/kfold_<orig-label>.json
#   - Checkpts: checkpoints/ckpt_kfold_<label>_fold<N>of5_best.pth
#
# Hardware target: RTX 3090 (24 GB VRAM) + 12-core CPU, 18 GB system RAM.
# Concurrency: exactly 3 parallel Python processes at all times (MAX_JOBS=3).
#
# Usage:
#   chmod +x scripts/run_folds.sh
#   ./scripts/run_folds.sh --dry-run 01 03 05 13 17 18 20 23 24 08 25
#   ./scripts/run_folds.sh 01 03 05 13 17 18 20 23 24 08 25
#
#   <run-id> ... : explicit run-id list (1..26), resolved against the SAME
#                  run definitions as run_all_experiments.sh.
#   --dry-run    : print the planned invocations only; launch nothing.
###############################################################################
set -euo pipefail

# Define colors for warnings
RED='\033[0;31m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Environment activation
# ---------------------------------------------------------------------------
echo "============================================================"
echo " $(date '+%Y-%m-%d %H:%M:%S') - Activating virtual environment ..."
echo "============================================================"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/venv/bin/activate"
echo " $(date '+%Y-%m-%d %H:%M:%S') - Virtual environment active: $(which python)"

# Limit intra-op threads per process to prevent CPU oversubscription across concurrent jobs
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

# ---------------------------------------------------------------------------
# Pre-flight Hardware & VRAM Verification (same threshold as run_all_experiments.sh)
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | awk '{print $1}')
    TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | awk '{print $1}')
    echo " $(date '+%Y-%m-%d %H:%M:%S') - GPU VRAM check: ${FREE_VRAM} MiB free / ${TOTAL_VRAM} MiB total"
    if [ -n "$FREE_VRAM" ] && [ "$FREE_VRAM" -lt 12000 ]; then
        echo -e "${RED}====================================================================${NC}"
        echo -e "${RED} WARNING: Insufficient free VRAM detected (${FREE_VRAM} MiB < 12000 MiB required).${NC}"
        echo -e "${RED} Another process (e.g., local LLM / vLLM) may be occupying the GPU.${NC}"
        echo -e "${RED} Please ensure GPU memory is freed before starting 3-way concurrent training.${NC}"
        echo -e "${RED}====================================================================${NC}"
    fi
fi

# ---------------------------------------------------------------------------
# Concurrency control (max 3 parallel Python processes)
# ---------------------------------------------------------------------------
MAX_JOBS=3
declare -a PIDS=()

# ---------------------------------------------------------------------------
# Run definitions (GROUNDED in run_all_experiments.sh / aggregate_results.py)
# ---------------------------------------------------------------------------
# Each entry: "<run_id> <run_name> <main.py flags...>"
# The flags are byte-for-byte the per-run flags from run_all_experiments.sh.
# The 5-fold driver appends: --k-folds 5 --summary-out ... --run-label kfold_<name>
#
# NOTE: run 18 (g4_panda_isolate_gn) uses --enable-gradnorm (GradNorm ON) and
# therefore must NOT carry --compile (main.py auto-disables compile when
# GradNorm is active). This matches run_all_experiments.sh exactly.
RUN_DEFS=(
    "01 g1_tcga_vgg16 --phase v1 --datasets tcga --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --compile --num-workers 2"
    "02 g1_tcga_mobilenet_v2 --phase v1 --datasets tcga --encoders mobilenet_v2 --no-macenko --disable-gradnorm --static-weights --num-workers 2"
    "03 g1_panda_vgg16 --phase v1 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --compile --num-workers 2"
    "04 g1_panda_mobilenet_v2 --phase v1 --datasets panda --encoders mobilenet_v2 --no-macenko --disable-gradnorm --static-weights --num-workers 2"
    "05 g1_siim_vgg16 --phase v1 --datasets siim --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --compile --num-workers 2"
    "06 g1_siim_mobilenet_v2 --phase v1 --datasets siim --encoders mobilenet_v2 --no-macenko --disable-gradnorm --static-weights --num-workers 2"
    "07 g2_tcga_vgg16 --phase v2 --datasets tcga --encoders vgg16 --num-workers 2"
    "08 g2_tcga_mobilenet_v2 --phase v2 --datasets tcga --encoders mobilenet_v2 --num-workers 2"
    "09 g2_panda_vgg16 --phase v2 --datasets panda --encoders vgg16 --num-workers 2"
    "10 g2_panda_mobilenet_v2 --phase v2 --datasets panda --encoders mobilenet_v2 --num-workers 2"
    "11 g2_siim_vgg16 --phase v2 --datasets siim --encoders vgg16 --num-workers 2"
    "12 g2_siim_mobilenet_v2 --phase v2 --datasets siim --encoders mobilenet_v2 --num-workers 2"
    "13 g3_pannuke_vgg16_naked --phase v1 --datasets pannuke --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --compile --num-workers 2"
    "14 g3_pannuke_mobilenet_v2_naked --phase v1 --datasets pannuke --encoders mobilenet_v2 --no-macenko --disable-gradnorm --static-weights --compile --num-workers 2"
    "15 g3_pannuke_vgg16_final --phase v2 --datasets pannuke --encoders vgg16 --num-workers 2"
    "16 g3_pannuke_mobilenet_v2_final --phase v2 --datasets pannuke --encoders mobilenet_v2 --num-workers 2"
    "17 g4_panda_isolate_lr --phase v2 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --num-workers 2"
    "18 g4_panda_isolate_gn --phase v1 --datasets panda --encoders vgg16 --no-macenko --enable-gradnorm --num-workers 2"
    "19 g4_panda_lambda_1_1 --phase v1 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --lambda-seg 1 --lambda-cls 1 --compile --num-workers 2"
    "20 g4_panda_lambda_5_1 --phase v1 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --lambda-seg 5 --lambda-cls 1 --compile --num-workers 2"
    "21 g4_panda_lambda_1_10 --phase v1 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --lambda-seg 1 --lambda-cls 10 --compile --num-workers 2"
    "22 g4_panda_lambda_10_1 --phase v1 --datasets panda --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --lambda-seg 10 --lambda-cls 1 --compile --num-workers 2"
    "23 g5_panda_nomacenko --phase v2 --datasets panda --encoders mobilenet_v2 --no-macenko --num-workers 2"
    "24 g5_pannuke_nomacenko --phase v2 --datasets pannuke --encoders mobilenet_v2 --no-macenko --num-workers 2"
    "25 g5_tcga_noskip --phase v2 --datasets tcga --encoders mobilenet_v2 --no-skip-connections --num-workers 2"
    "26 g5_panda_noskip --phase v2 --datasets panda --encoders mobilenet_v2 --no-skip-connections --num-workers 2"
)

# ---------------------------------------------------------------------------
# Argument parsing: [--dry-run] <run-id>...
# ---------------------------------------------------------------------------
DRY_RUN=0
RUN_IDS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=1
            ;;
        *)
            RUN_IDS+=("$arg")
            ;;
    esac
done

if [ "${#RUN_IDS[@]}" -eq 0 ]; then
    echo -e "${RED}ERROR: no run-id(s) supplied.${NC}"
    echo "Usage: $0 [--dry-run] <run-id>...   (run-id in 01..26)"
    echo "Example: $0 --dry-run 01 03 05 13 17 18 20 23 24 08 25"
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve run-id -> definition
# ---------------------------------------------------------------------------
lookup_def() {
    local want="$1"
    local entry
    for entry in "${RUN_DEFS[@]}"; do
        local id name
        id=$(printf '%s' "$entry" | awk '{print $1}')
        name=$(printf '%s' "$entry" | awk '{print $2}')
        if [ "$id" = "$want" ]; then
            printf '%s' "$entry"
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Build the full command for a resolved run definition
# ---------------------------------------------------------------------------
# $1 = run_id, $2 = run_name, $3 = flags string
build_cmd() {
    local run_id="$1"
    local run_name="$2"
    local flags="$3"
    local padded_id
    padded_id=$(printf '%02d' $((10#${run_id})))
    local run_label="kfold_${run_name}"
    local summary_file="results/round2/kfold_${run_name}.json"

    # shellcheck disable=SC2206
    local cmd=(python main.py ${flags} --k-folds 5 --no-resume --summary-out "${summary_file}" --run-label "${run_label}")
    printf '%s\n' "${cmd[*]}"
}

# ---------------------------------------------------------------------------
# Safe Auto-Healing Wrapper (mirrors run_all_experiments.sh)
# ---------------------------------------------------------------------------
# Retries a failed run with reduced system-level optimizations to handle
# transient OOM, torch.compile crashes, or DataLoader worker spikes.
# Does NOT alter any hyperparameters (batch size, LR, epochs, loss weights).
run_with_safe_healing() {
    local run_id="$1"
    local run_name="$2"
    local log_file="$3"
    shift 3
    local cmd=("$@")

    # ── Attempt 1: Standard run ──────────────────────────────────────────
    {
        echo "[$run_id] $(date '+%Y-%m-%d %H:%M:%S') - [ATTEMPT 1/2] Starting standard 5-fold run..."
        echo "[$run_id] Command: ${cmd[*]}"
    } >> "$log_file" 2>&1

    local rc=0
    "${cmd[@]}" >> "$log_file" 2>&1 || rc=$?

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
    "${cmd[@]}" >> "$log_file" 2>&1 || rc=$?

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
# Launch a single 5-fold run (background)
# ---------------------------------------------------------------------------
launch_job() {
    local run_id="$1"
    local run_name="$2"
    local flags="$3"

    local padded_id
    padded_id=$(printf '%02d' $((10#${run_id})))
    local run_label="kfold_${run_name}"
    local log_file="logs/kfold_${run_name}.log"
    local summary_file="results/round2/kfold_${run_name}.json"

    mkdir -p logs results/round2

    echo " [${run_id}] $(date '+%Y-%m-%d %H:%M:%S') - START (5-fold): ${run_name}"
    echo "        Log:      ${log_file}"
    echo "        Summary: ${summary_file}"

    run_with_safe_healing "$run_id" "$run_name" "$log_file" \
        python main.py ${flags} --k-folds 5 --no-resume \
        --summary-out "${summary_file}" --run-label "${run_label}" >> /dev/null 2>&1 &
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

wait_all() {
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    PIDS=()
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
echo "============================================================"
echo " $(date '+%Y-%m-%d %H:%M:%S') - 5-FOLD CAMPAIGN (k=5, group-aware)"
echo "============================================================"
echo " Requested run-ids: ${RUN_IDS[*]}"
echo " Concurrency:       ${MAX_JOBS} parallel processes"
echo "============================================================"

# Resolve all requested ids up-front (fail fast on unknown ids)
declare -a RESOLVED=()
for rid in "${RUN_IDS[@]}"; do
    local_padded=$(printf '%02d' $((10#${rid})))
    def=$(lookup_def "$local_padded") || {
        echo -e "${RED}ERROR: unknown run-id '${rid}' (expected 01..26).${NC}"
        exit 1
    }
    RESOLVED+=("$def")
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo " [DRY-RUN] Planned 5-fold invocations (nothing will be launched):"
    echo " ----------------------------------------------------------------"
    for def in "${RESOLVED[@]}"; do
        read -r rid rname rflags <<< "$def"
        cmd=$(build_cmd "$rid" "$rname" "$rflags")
        echo "  [${rid}] ${cmd}"
    done
    echo " ----------------------------------------------------------------"
    echo " [DRY-RUN] ${#RESOLVED[@]} run(s) planned. Exiting without launching."
    exit 0
fi

for def in "${RESOLVED[@]}"; do
    read -r rid rname rflags <<< "$def"
    wait_for_slot
    launch_job "$rid" "$rname" "$rflags"
done

echo " $(date '+%Y-%m-%d %H:%M:%S') - All ${#RESOLVED[@]} 5-fold jobs launched. Waiting for completion ..."
wait_all

echo "============================================================"
echo " $(date '+%Y-%m-%d %H:%M:%S') - 5-FOLD CAMPAIGN COMPLETE"
echo "============================================================"
echo " Logs:      logs/kfold_*.log"
echo " Summaries: results/round2/kfold_*.json"
echo " Checkpts:  checkpoints/ckpt_kfold_*_fold*of5_best.pth"
