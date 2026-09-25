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
#   - Summary:  results/kfold_campaign/kfold_<orig-label>.json
#   - Checkpts: results/kfold_campaign/kfold_<label>_fold<N>of5/best.pt
#               results/kfold_campaign/kfold_<label>_fold<N>of5/final.state.pt
#               (deterministic per-fold paths; final.state.pt is the resume
#               state, written atomically. The old checkpoints/ckpt_kfold_*_
#               best.pth naming is gone.)
#
# Resume (power-cut recovery):
#   FOLD_RESUME=1 ./scripts/run_folds.sh <run-id>...
#     -> appends --resume: each fold resumes from its last completed epoch
#        (final.state.pt) instead of retraining from scratch.
#   FOLD_RESUME unset/0 (DEFAULT) -> appends --no-resume: clean-room fresh
#        start, prior state ignored (the original campaign behavior).
#   A resumed fold is FATAL if its state's fingerprint (code/config/torch
#   version) does not match the current run; set RESUME_ALLOW_LEGACY=1 to
#   bridge pre-hardening states (loud WARNING, no identity check).
#
# Hardware target: RTX 3090 (24 GB VRAM) + 12-core CPU, 20 GB system RAM (14 GB free).
# Concurrency: up to 3 parallel Python processes (MAX_JOBS=3), gated by a
# VRAM-aware defer-not-abort admission (see vram_admission_ok below).
#
# Usage:
#   chmod +x scripts/run_folds.sh
#   ./scripts/run_folds.sh --dry-run 01 03 05 13 17 18 20 23 24 08 25
#   ./scripts/run_folds.sh 01 03 05 13 17 18 20 23 24 08 25
#   FOLD_RESUME=1 ./scripts/run_folds.sh 01 03 05   # resume after a power cut
#
#   <run-id> ... : explicit run-id list (1..27), resolved against the SAME
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
# Pin the visible GPU (default: device 0) so the campaign never races onto a busy device
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ---------------------------------------------------------------------------
# Pre-flight Hardware & VRAM Verification (same threshold as run_all_experiments.sh)
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | awk '{print $1}')
    TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | awk '{print $1}')
    echo " $(date '+%Y-%m-%d %H:%M:%S') - GPU VRAM check: ${FREE_VRAM} MiB free / ${TOTAL_VRAM} MiB total"
    if [ -n "$FREE_VRAM" ] && [ "$FREE_VRAM" -lt 10240 ]; then
        echo -e "${RED}====================================================================${NC}"
        echo -e "${RED} FATAL: Insufficient free VRAM (${FREE_VRAM} MiB < 10240 MiB / 10 GiB required).${NC}"
        echo -e "${RED} Another process (e.g., local LLM / vLLM) is occupying the GPU.${NC}"
        echo -e "${RED} Aborting before launch — free GPU memory and re-run.${NC}"
        echo -e "${RED}====================================================================${NC}"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Concurrency control (max 2 parallel (env-overridable) Python processes, VRAM-admission-gated)
# ---------------------------------------------------------------------------
MAX_JOBS="${MAX_JOBS:-2}"
declare -a PIDS=()
declare -A PID_PRED_VRAM=()   # pid -> predicted VRAM (MiB), for admission

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
    # Run 27: canonical GradNorm arm (Chen et al. 2018). --gradnorm-mode
    # canonical selects the NATIVE canonical implementation in src/training.py
    # (raw task-weight parameters in a DEDICATED Adam optimizer, strict
    # gradient detachment, per-step renormalization to sum=2) — mirroring the
    # standalone probe scripts/run_canonical_gradnorm_panda.py L151-224.
    # --enable-gradnorm turns GradNorm on; --gradnorm-alpha 1.5 is the probe's
    # alpha. Full epochs (main.py default), 5 folds (driver-appended
    # --k-folds 5). No --compile (GradNorm active).
    "27 canonical_gradnorm --phase v1 --datasets panda --encoders vgg16 --no-macenko --enable-gradnorm --gradnorm-alpha 1.5 --gradnorm-mode canonical --num-workers 2"
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
    echo "Usage: $0 [--dry-run] <run-id>...   (run-id in 01..27)"
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
    local summary_file="results/kfold_campaign/kfold_${run_name}.json"

    # FOLD_RESUME=1 -> --resume (power-cut recovery); default -> --no-resume
    # (clean-room fresh start, the original campaign behavior).
    local resume_flag="--no-resume"
    if [ "${FOLD_RESUME:-0}" = "1" ]; then
        resume_flag="--resume"
    fi

    # shellcheck disable=SC2206
    local cmd=(python main.py ${flags} --k-folds 5 ${resume_flag} --summary-out "${summary_file}" --run-label "${run_label}")
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
    local summary_file="results/kfold_campaign/kfold_${run_name}.json"

    # FOLD_RESUME=1 -> --resume (power-cut recovery); default -> --no-resume
    # (clean-room fresh start, the original campaign behavior).
    local resume_flag="--no-resume"
    if [ "${FOLD_RESUME:-0}" = "1" ]; then
        resume_flag="--resume"
    fi

    mkdir -p logs results/kfold_campaign

    echo " [${run_id}] $(date '+%Y-%m-%d %H:%M:%S') - START (5-fold): ${run_name}"
    echo "        Log:      ${log_file}"
    echo "        Summary: ${summary_file}"
    echo "        Resume:   ${resume_flag}"

    run_with_safe_healing "$run_id" "$run_name" "$log_file" \
        python main.py ${flags} --k-folds 5 ${resume_flag} \
        --summary-out "${summary_file}" --run-label "${run_label}" >> /dev/null 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    local dataset encoder
    dataset=$(printf '%s' "$flags" | awk '{for(i=1;i<NF;i++) if($i=="--datasets"){print $(i+1); exit}}')
    encoder=$(printf '%s' "$flags" | awk '{for(i=1;i<NF;i++) if($i=="--encoders"){print $(i+1); exit}}')
    PID_PRED_VRAM[$pid]=$(vram_predict "$dataset" "$encoder")
    echo " [${run_id}] PID: ${pid} (predicted VRAM ${PID_PRED_VRAM[$pid]} MiB)"
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

# ---------------------------------------------------------------------------
# RAM watchdog (ported from scripts/wave_scheduler.py, defer-not-kill semantics)
# WSL has 20 GB total / ~14 GB free: per-job budget = soft cap / MAX_JOBS.
# A job breaching its budget is aborted ALONE; the campaign continues.
# ---------------------------------------------------------------------------
GLOBAL_RAM_SOFT_CAP_GB=17   # 20 GB WSL RAM - headroom (matches wave_scheduler.py)
RAM_BUDGET_KB=$(( GLOBAL_RAM_SOFT_CAP_GB * 1024 * 1024 / MAX_JOBS ))

ram_watchdog() {
    local total_kb=0 pid rss
    for pid in "${PIDS[@]}"; do
        rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ') || rss=""
        if [ -z "$rss" ]; then
            continue
        fi
        total_kb=$(( total_kb + rss ))
        if [ "$rss" -gt "$RAM_BUDGET_KB" ]; then
            echo " [RAM-WATCHDOG] $(date '+%Y-%m-%d %H:%M:%S') - pid=${pid} RSS $(( rss / 1024 )) MB > budget $(( RAM_BUDGET_KB / 1024 )) MB; aborting this job only (campaign continues)"
            kill "$pid" 2>/dev/null || true
        fi
    done
    if [ "$total_kb" -gt $(( GLOBAL_RAM_SOFT_CAP_GB * 1024 * 1024 )) ]; then
        echo " [RAM-WATCHDOG] WARNING: global RAM soft cap breached ($(( total_kb / 1024 / 1024 )) GB > ${GLOBAL_RAM_SOFT_CAP_GB} GB)"
    fi
}

# ---------------------------------------------------------------------------
# VRAM-aware admission (ported from scripts/wave_scheduler.py: predict_vram
# L169-175, query_vram L209-226, defer-not-abort admission in run_wave L490-512)
# Per-job peak VRAM measured in r31 (MiB); mobilenet_v2 ≈ 60% of the vgg16 peak.
# ---------------------------------------------------------------------------
VRAM_BUDGET_MIB=21000   # 24576 MiB card - ~1.5 GB headroom over desktop baseline

vram_predict() {
    local dataset="$1" encoder="$2"
    case "$encoder" in
        vgg16)
            case "$dataset" in
                tcga)    echo 8816 ;;
                pannuke) echo 8429 ;;
                siim)    echo 8425 ;;
                panda)   echo 3205 ;;
                *)       echo 5500 ;;
            esac ;;
        mobilenet_v2)
            case "$dataset" in
                tcga)    echo 5290 ;;   # 60% of 8816
                pannuke) echo 5057 ;;   # 60% of 8429
                siim)    echo 5055 ;;   # 60% of 8425
                panda)   echo 1923 ;;   # 60% of 3205
                *)       echo 3300 ;;
            esac ;;
        *) echo 5500 ;;
    esac
}

# Predicted VRAM (MiB) for one live job pid; worst-case 8816 when unknown.
# WSL nvidia-smi reports "[N/A]" per-process memory, so we admit on PREDICTIONS
# only (r31-measured peaks; conservative by construction).
vram_pred_of_pid() {
    local pid="$1" pred
    case "${PID_PRED_VRAM[$pid]:-}" in
        ''|*[!0-9]*) echo 8816 ;;
        *) echo "${PID_PRED_VRAM[$pid]}" | tr -d '[:space:]' ;;
    esac
}

# Defer-not-abort: return 0 only if the new job's predicted VRAM fits the
# budget alongside LIVE running jobs (zombie-safe; integer-validated sum).
vram_admission_ok() {
    local new_pred="$1" total=0 pid used active=0
    case "$new_pred" in ''|*[!0-9]*) new_pred=8816 ;; esac
    for pid in "${PIDS[@]}"; do
        local stat
        stat=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]')
        case "$stat" in ''|Z*) continue ;; esac   # gone or zombie: skip
        active=$(( active + 1 ))
        used=$(vram_pred_of_pid "$pid")
        case "$used" in ''|*[!0-9]*) used=8816 ;; esac
        total=$(( total + used ))
    done
    VRAM_ADMIT_TOTAL="$total"; VRAM_ADMIT_ACTIVE="$active"
    [ $(( total + new_pred )) -le "$VRAM_BUDGET_MIB" ]
}

wait_for_slot() {
    local new_pred="$1"
    case "$new_pred" in ''|*[!0-9]*) new_pred=8816 ;; esac
    while true; do
        local active
        active=$(count_active)
        ram_watchdog
        if [ "$active" -lt "$MAX_JOBS" ] && vram_admission_ok "$new_pred"; then
            break
        fi
        if [ "$active" -lt "$MAX_JOBS" ]; then
            echo " [VRAM-ADMISSION] $(date '+%Y-%m-%d %H:%M:%S') - deferring next job (predicted ${new_pred} MiB; budget ${VRAM_BUDGET_MIB} MiB); live=${VRAM_ADMIT_ACTIVE:-?} total=${VRAM_ADMIT_TOTAL:-?}+${new_pred}; live pids: $(printf '%s ' "${PIDS[@]:-}" 2>/dev/null); re-checking in 20s"
        fi
        sleep 20
    done
}

wait_all() {
    # wait -n (bash >= 4.3) reaps one job at a time so the RAM watchdog
    # stays live for the whole campaign, not just the launch phase.
    while [ "${#PIDS[@]}" -gt 0 ]; do
        wait -n "${PIDS[@]}" 2>/dev/null || true
        ram_watchdog
        count_active >/dev/null
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
        echo -e "${RED}ERROR: unknown run-id '${rid}' (expected 01..27).${NC}"
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
    local_dataset=$(printf '%s' "$rflags" | awk '{for(i=1;i<NF;i++) if($i=="--datasets"){print $(i+1); exit}}')
    local_encoder=$(printf '%s' "$rflags" | awk '{for(i=1;i<NF;i++) if($i=="--encoders"){print $(i+1); exit}}')
    wait_for_slot "$(vram_predict "$local_dataset" "$local_encoder")"
    launch_job "$rid" "$rname" "$rflags"
done

echo " $(date '+%Y-%m-%d %H:%M:%S') - All ${#RESOLVED[@]} 5-fold jobs launched. Waiting for completion ..."
wait_all

# Post-campaign aggregation: populate fold/fold_sd columns in dice_ci_summary.csv
echo " $(date '+%Y-%m-%d %H:%M:%S') - Post-campaign: python scripts/bootstrap_dice_ci.py --fold-aware"
python scripts/bootstrap_dice_ci.py --fold-aware || \
    echo " WARNING: bootstrap_dice_ci.py --fold-aware failed (campaign results are intact)"

echo "============================================================"
echo " $(date '+%Y-%m-%d %H:%M:%S') - 5-FOLD CAMPAIGN COMPLETE"
echo "============================================================"
echo " Logs:      logs/kfold_*.log"
echo " Summaries: results/kfold_campaign/kfold_*.json"
echo " Checkpts:  checkpoints/ckpt_kfold_*_fold*of5_best.pth"
