#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# R3-1 GPU SMOKE GATE
# ---------------------------------------------------------------------------
# Fail-fast, unattended-safe pre-flight gate for the 26-run + 130-fold
# campaign (run_all_experiments.sh). Validates, in order:
#
#   (a) Parser dry-load of all 4 datasets with hard row-count asserts
#       (tcga=3929, panda=10516, siim=10675, pannuke=7901), including the
#       Macenko image paths (panda/pannuke default Macenko ON).
#   (b) One REAL 2-epoch training run per dataset via main.py, mirroring
#       run_all_experiments.sh flags (--phase v2, --num-workers 2). The
#       HEAVIEST campaign encoder (vgg16) is used for every dataset so the
#       VRAM probe reflects the worst case MAX_JOBS must be derived from.
#   (c) Artifact gate: scripts/assert_artifacts.py on results/round2/<label>/
#       (epoch_log.jsonl fields, 10-field per_slice_dice.jsonl, best.pt,
#       final.state.pt, summary resume_branch/resumed_from_epoch).
#   (d) Cross-process resume check: 1-epoch run, then rerun with the same
#       --run-label at --epochs 2 -> assert resumed_from_epoch==1.
#   (e) VRAM (nvidia-smi poll loop) + epoch-time sampling -> recommended
#       MAX_JOBS and full-campaign wall-time estimate (26 single-split +
#       130 fold-runs, 50 epochs each).
#
# Usage:
#     bash run_smoke_test.sh            # real GPU run (fail-fast)
#     bash run_smoke_test.sh --dry-run  # print planned invocations only
#
# Any failure aborts immediately (set -e + explicit FATAL exits).
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Argument parsing ───────────────────────────────────────────────────────
DRY_RUN=0
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            grep '^#' "$0" | head -30
            exit 0 ;;
        *)
            echo "FATAL: unknown argument: $a (supported: --dry-run)" >&2
            exit 1 ;;
    esac
done

# ── Virtual environment ────────────────────────────────────────────────────
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
else
    echo "FATAL: venv/ not found at $SCRIPT_DIR/venv" >&2
    exit 1
fi

# ── Thread limits (mirror run_all_experiments.sh; conservative for WSL) ───
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

# ── Gate constants ─────────────────────────────────────────────────────────
# Heaviest campaign encoder: run_all_experiments.sh runs vgg16 AND
# mobilenet_v2 for every dataset; VGG16 is the heavier backbone (params,
# activations, VRAM), so the VRAM probe must use it.
ENCODER="vgg16"
NUM_WORKERS=2          # conservative DataLoader workers (WSL 10GB RAM cap)
EPOCHS=2
CAMPAIGN_EPOCHS=50     # phase v2 default
DATASETS=(tcga panda siim pannuke)
# Single-split run counts per dataset in the 26-run matrix (g1+g2+g4+g5):
declare -A SINGLE_SPLIT_COUNT=( [tcga]=5 [panda]=12 [siim]=4 [pannuke]=5 )
# Expected parser row counts (hard contract)
declare -A EXPECTED_ROWS=( [tcga]=3929 [panda]=10516 [siim]=10675 [pannuke]=7901 )

LOG_DIR="smoke_test_logs/r31"
mkdir -p "$LOG_DIR"

fail() {
    echo "FATAL: $*" >&2
    echo "FATAL: R3-1 gate aborted at $(date '+%Y-%m-%d %H:%M:%S') (log dir: $LOG_DIR)" >&2
    exit 1
}

# run_or_print CMD... : execute in real mode, print in --dry-run mode
run_or_print() {
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY-RUN] $*"
    else
        echo "[RUN] $*"
        "$@"
    fi
}

echo "============================================================"
echo " R3-1 GPU SMOKE GATE  ($(date '+%Y-%m-%d %H:%M:%S'))"
if [ "$DRY_RUN" -eq 1 ]; then MODE="DRY-RUN (planned invocations only)"; else MODE="REAL (fail-fast)"; fi
echo " Mode: $MODE"
echo " Encoder probe: $ENCODER (heaviest campaign backbone)"
echo "============================================================"

# ── Preflight: GPU availability (real mode only) ───────────────────────────
if [ "$DRY_RUN" -eq 0 ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        fail "nvidia-smi not found; R3-1 is a GPU gate (use --dry-run for CPU authoring)"
    fi
    FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    echo " GPU: ${FREE_VRAM} MiB free / ${TOTAL_VRAM} MiB total"
    if [ "$FREE_VRAM" -lt 8000 ]; then
        fail "insufficient free VRAM (${FREE_VRAM} MiB < 8000 MiB); another process may hold the GPU"
    fi
else
    echo "[DRY-RUN] preflight: nvidia-smi VRAM check (free >= 8000 MiB)"
fi

# ===========================================================================
# (a) Parser dry-load: all 4 datasets, hard row-count asserts, Macenko paths
# ===========================================================================
echo ""
echo ">> (a) Parser dry-load with row-count asserts"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] python: load_dataset_bundle(tcga/panda/siim/pannuke, Macenko paths) asserting rows tcga=3929 panda=10516 siim=10675 pannuke=7901"
else
    python - <<'PY'
import sys
from src.config import DATASET_ROOTS
from src.data import load_dataset_bundle

expected = {"tcga": 3929, "panda": 10516, "siim": 10675, "pannuke": 7901}
for ds in ("tcga", "panda", "siim", "pannuke"):
    # skip_macenko=False -> default Macenko image paths for panda/pannuke
    bundle = load_dataset_bundle(ds, DATASET_ROOTS[ds], skip_macenko=False)
    n = len(bundle["images"])
    if n != expected[ds]:
        print(f"FATAL: {ds} row count {n} != expected {expected[ds]}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {ds:8s} rows={n} (expected {expected[ds]})")
print("PASS: all 4 datasets parse with exact expected row counts")
PY
fi

# ===========================================================================
# (b) Per-dataset 2-epoch real training runs (heaviest encoder: vgg16)
#     (c) Artifact gate per run
# ===========================================================================
echo ""
echo ">> (b)+(c) Per-dataset 2-epoch runs (vgg16) + artifact gate"
for ds in "${DATASETS[@]}"; do
    label="r31_${ds}_${ENCODER}"
    summary="checkpoints/summary_${label}.json"
    log="$LOG_DIR/train_${label}.log"
    vram_log="$LOG_DIR/vram_${label}.log"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY-RUN] python main.py --phase v2 --datasets $ds --encoders $ENCODER --epochs $EPOCHS --num-workers $NUM_WORKERS --no-resume --run-label $label --summary-out $summary"
        echo "[DRY-RUN] python scripts/assert_artifacts.py --run-label $label --summary $summary"
        continue
    fi

    # VRAM poll loop (2s) during this run
    : > "$vram_log"
    (
        while true; do
            nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits >> "$vram_log" 2>/dev/null || break
            sleep 2
        done
    ) &
    poller=$!

    echo " [$(date '+%H:%M:%S')] START $ds x $ENCODER (2 epochs, workers=$NUM_WORKERS)"
    if ! python main.py --phase v2 --datasets "$ds" --encoders "$ENCODER" \
            --epochs "$EPOCHS" --num-workers "$NUM_WORKERS" --no-resume \
            --run-label "$label" --summary-out "$summary" > "$log" 2>&1; then
        kill "$poller" 2>/dev/null || true
        echo "  Last 30 lines of $log:" >&2
        tail -n 30 "$log" >&2
        fail "training run $label failed"
    fi
    kill "$poller" 2>/dev/null || true
    wait "$poller" 2>/dev/null || true

    # (c) artifact gate
    if ! python scripts/assert_artifacts.py --run-label "$label" --summary "$summary"; then
        fail "artifact gate failed for $label"
    fi
    echo " [$(date '+%H:%M:%S')] PASS $ds x $ENCODER"
done

# ===========================================================================
# (d) Cross-process resume check: 1-epoch run, then rerun --epochs 2
# ===========================================================================
echo ""
echo ">> (d) Cross-process resume check (tcga x vgg16)"
R_LABEL="r31_resume_tcga"
R_SUMMARY="checkpoints/summary_${R_LABEL}.json"
R_LOG="$LOG_DIR/train_${R_LABEL}.log"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] python main.py --phase v2 --datasets tcga --encoders $ENCODER --epochs 1 --num-workers $NUM_WORKERS --no-resume --run-label $R_LABEL --summary-out $R_SUMMARY"
    echo "[DRY-RUN] python main.py --phase v2 --datasets tcga --encoders $ENCODER --epochs 2 --num-workers $NUM_WORKERS --resume --run-label $R_LABEL --summary-out $R_SUMMARY"
    echo "[DRY-RUN] python scripts/assert_artifacts.py --run-label $R_LABEL --summary $R_SUMMARY --expect-resumed-from-epoch 1 --expect-resume-branch resumed-from-epoch"
else
    # Pass 1: fresh 1-epoch run (mint final.state.pt at epoch 1)
    if ! python main.py --phase v2 --datasets tcga --encoders "$ENCODER" \
            --epochs 1 --num-workers "$NUM_WORKERS" --no-resume \
            --run-label "$R_LABEL" --summary-out "$R_SUMMARY" > "$R_LOG" 2>&1; then
        tail -n 30 "$R_LOG" >&2
        fail "resume-check pass 1 (1 epoch) failed"
    fi
    # Pass 2: NEW process, same --run-label, --epochs 2 -> must resume from epoch 1
    if ! python main.py --phase v2 --datasets tcga --encoders "$ENCODER" \
            --epochs 2 --num-workers "$NUM_WORKERS" --resume \
            --run-label "$R_LABEL" --summary-out "$R_SUMMARY" >> "$R_LOG" 2>&1; then
        tail -n 30 "$R_LOG" >&2
        fail "resume-check pass 2 (resume to 2 epochs) failed"
    fi
    if ! python scripts/assert_artifacts.py --run-label "$R_LABEL" --summary "$R_SUMMARY" \
            --expect-resumed-from-epoch 1 --expect-resume-branch resumed-from-epoch; then
        fail "resume-check artifact assertions failed (expected resumed_from_epoch==1)"
    fi
    echo " PASS: cross-process resume verified (resumed_from_epoch==1)"
fi

# ===========================================================================
# (e) VRAM + epoch-time sampling -> recommended MAX_JOBS + campaign estimate
# ===========================================================================
echo ""
echo ">> (e) VRAM / epoch-time sampling -> MAX_JOBS + campaign estimate"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] python: sample peak VRAM from $LOG_DIR/vram_*.log + epoch_sec from results/round2/r31_*/epoch_log.jsonl; print recommended MAX_JOBS and 26+130-run estimate"
else
    python - <<'PY'
import json
import math
import os
import re
import subprocess
from pathlib import Path

log_dir = Path("smoke_test_logs/r31")
datasets = ("tcga", "panda", "siim", "pannuke")
single_split = {"tcga": 5, "panda": 12, "siim": 4, "pannuke": 5}  # 26 total
campaign_epochs = 50

# --- peak VRAM per dataset from the 2s nvidia-smi poll logs ---
peak_vram = {}
for ds in datasets:
    p = log_dir / f"vram_r31_{ds}_vgg16.log"
    peak = 0
    if p.exists():
        for line in p.read_text().splitlines():
            m = re.match(r"\s*(\d+)\s*,\s*(\d+)", line)
            if m:
                peak = max(peak, int(m.group(1)))
    peak_vram[ds] = peak
worst_vram = max(peak_vram.values()) if peak_vram else 0

# --- mean epoch time per dataset from per-run epoch logs ---
epoch_sec = {}
for ds in datasets:
    p = Path("results/round2") / f"r31_{ds}_vgg16" / "epoch_log.jsonl"
    vals = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec.get("epoch_sec"), (int, float)):
                vals.append(float(rec["epoch_sec"]))
    epoch_sec[ds] = (sum(vals) / len(vals)) if vals else None

# --- recommended MAX_JOBS ---
vram_jobs = cpu_jobs = ram_jobs = None
free_vram = total_vram = None
try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip().splitlines()[0].split(",")
    free_vram, total_vram = int(out[0].strip()), int(out[1].strip())
    if worst_vram > 0:
        vram_jobs = max(1, free_vram // worst_vram)
except Exception:
    pass
try:
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        parts = line.split()
        if parts[0].endswith(":"):
            mem[parts[0][:-1]] = float(parts[1])
    ram_jobs = max(1, int(mem.get("MemAvailable", 0) / 1024 / 1024 // 3))  # ~3GB/job
except Exception:
    pass
cpu_jobs = max(1, (os.cpu_count() or 2) // 2)  # 2 intra-op threads per job

candidates = [j for j in (vram_jobs, ram_jobs, cpu_jobs) if j]
max_jobs = max(1, min(candidates)) if candidates else 1

# --- full-campaign estimate: 26 single-split + 130 fold-runs, 50 epochs each ---
total_epochs = 0
missing = []
for ds in datasets:
    if epoch_sec[ds] is None:
        missing.append(ds)
        continue
    total_epochs += (single_split[ds] + 5 * single_split[ds]) * campaign_epochs

print("=" * 60)
print(" R3-1 VRAM / THROUGHPUT REPORT")
print("=" * 60)
for ds in datasets:
    print(f"  {ds:8s} peak_vram={peak_vram[ds]:>6} MiB  mean_epoch={epoch_sec[ds]}")
print(f"  worst-case peak VRAM (vgg16): {worst_vram} MiB")
if free_vram is not None:
    print(f"  GPU now: {free_vram} MiB free / {total_vram} MiB total")
print(f"  job budgets: vram={vram_jobs} ram={ram_jobs} cpu={cpu_jobs}")
print(f"  RECOMMENDED MAX_JOBS = {max_jobs}")
if missing:
    print(f"  WARNING: no epoch timing for {missing}; estimate uses available data only")
if total_epochs > 0:
    # mean epoch time across datasets that have data
    known = [epoch_sec[ds] for ds in datasets if epoch_sec[ds] is not None]
    mean_ep = sum(known) / len(known)
    serial_h = total_epochs * mean_ep / 3600.0
    parallel_h = serial_h / max_jobs
    print(f"  full campaign: 26 single-split + 130 fold-runs x {campaign_epochs} epochs "
          f"= {total_epochs} epochs")
    print(f"  estimate: {serial_h:.1f} h serial -> {parallel_h:.1f} h at MAX_JOBS={max_jobs}")
else:
    print("  WARNING: no epoch timings collected; cannot estimate campaign duration")
print("=" * 60)
PY
fi

echo ""
echo "R3-1 gate $([ "$DRY_RUN" -eq 1 ] && echo 'dry-run complete (no GPU work executed)' || echo 'PASSED')"
exit 0
