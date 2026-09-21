#!/usr/bin/env bash
###############################################################################
# run_wave.sh
# Scaling campaign runner: 2-5 concurrent training jobs, GPU+CPU saturated,
# RAM-safe — measured, not guessed. Thin driver over scripts/wave_scheduler.py.
#
# Hardware target: WSL, 20 GB RAM, 8 procs, 2 GB swap, RTX 3090 24 GB.
#
# Features (see scripts/wave_scheduler.py for the full design):
#   1. VRAM-aware admission: R3-1 measured table (vgg16: tcga 8816, pannuke
#      8429, siim 6635, panda 3205 MiB) x per-encoder factor; live free VRAM
#      minus 1.5 GB safety margin. Non-fitting jobs DEFER (re-checked each
#      cycle), never aborted.
#   2. RAM watchdog: per-job RSS sampled each cycle; budget breach => that
#      job hard-aborted alone (wave continues); global soft cap ~17 GB logged.
#   3. CPU saturation: adaptive dataloader workers per job (1 job->6,
#      2->3, 3-4->2, 5->1 of 8 procs); persistent workers / prefetch follow
#      src/loader_tuning.py; requested value recorded in per-run log.
#   4. Mix ordering: run-list sorted by predicted VRAM, snake-ordered
#      (heavy/light alternation) so the concurrent window mixes big + small.
#   5. --max-jobs CLI cap (default 3, max 5); every admission decision logged
#      to logs/wave_scheduler.log (admitted/deferred + why).
#   6. Resume-safe + restartable: runs whose summary JSON exists are skipped;
#      runs launch with --resume (NEVER --no-resume); per-run log
#      logs/<label>.log; power-cut recovery = rerun the same command.
#   7. --calibrate N: run first N wave jobs with --epochs 2, sample
#      VRAM/RAM/epoch-time per concurrent slot, print recommended MAX_JOBS
#      + projected wave wall-time.
#
# Usage:
#   chmod +x scripts/run_wave.sh
#   ./scripts/run_wave.sh --dry-run 01 03 05 13 17 18 20 10 23 16 24 08 25
#   ./scripts/run_wave.sh --calibrate 3 01 03 05 13 17 18 20 10 23 16 24 08 25
#   ./scripts/run_wave.sh --max-jobs 4 01 03 05 13 17 18 20 10 23 16 24 08 25
#
#   <run-id> ... : explicit run-id list (01..26), resolved against the SAME
#                  run definitions as run_all_experiments.sh (single-split
#                  flags mirror it exactly; --num-workers is set adaptively).
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Environment activation
# ---------------------------------------------------------------------------
echo "============================================================"
echo " $(date '+%Y-%m-%d %H:%M:%S') - Activating virtual environment ..."
echo "============================================================"
source "${PROJECT_ROOT}/venv/bin/activate"
echo " $(date '+%Y-%m-%d %H:%M:%S') - Virtual environment active: $(which python)"

# Limit intra-op threads per process to prevent CPU oversubscription across
# concurrent jobs (8 procs shared by up to 5 jobs).
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

# ---------------------------------------------------------------------------
# Pre-flight hardware check (informational; admission is per-job in the
# scheduler, not a global gate).
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | awk '{print $1}')
    TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | awk '{print $1}')
    echo " $(date '+%Y-%m-%d %H:%M:%S') - GPU VRAM: ${FREE_VRAM} MiB free / ${TOTAL_VRAM} MiB total"
fi

mkdir -p logs

# ---------------------------------------------------------------------------
# Dispatch to the scheduler (all admission/watchdog logic lives there).
# ---------------------------------------------------------------------------
exec python "${SCRIPT_DIR}/wave_scheduler.py" "$@"
