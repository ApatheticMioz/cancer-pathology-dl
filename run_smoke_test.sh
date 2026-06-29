#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Exhaustive Smoke Test Suite (Refactored Module Coverage)
# ---------------------------------------------------------------------------
# Validates every unique Phase/Encoder/Ablation/Flag combination and verifies
# all refactored module imports before committing to the 26-run concurrent
# orchestrator for Q1 journal submission.
#
# Each test gets an isolated log file in smoke_test_logs/.
# On failure, the last 20 lines of the offending log are dumped to console
# so the developer instantly sees the Python traceback.
#
# Usage:
#     bash run_smoke_test.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Virtual environment ───────────────────────────────────────────────────
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
else
    echo -e "${RED}ERROR: venv/ not found at $SCRIPT_DIR/venv${NC}"
    exit 1
fi

# ── Log directory ──────────────────────────────────────────────────────────
LOG_DIR="$SCRIPT_DIR/smoke_test_logs"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

# ── Counters ───────────────────────────────────────────────────────────────
PASS=0
FAIL=0
TEST_INDEX=0

# ── Helpers ────────────────────────────────────────────────────────────────
banner() {
    local test_name="$1"
    local cmd="$2"
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  SMOKE TEST #$TEST_INDEX: $test_name${NC}"
    echo -e "${BLUE}  Command: $cmd${NC}"
    echo -e "${BLUE}  Log:   $LOG_DIR/$(printf "%02d" $TEST_INDEX)_${test_name// /_}.log${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

run_test() {
    local test_name="$1"
    local cmd="$2"
    local safe_name
    safe_name=$(echo "$test_name" | sed 's/ /_/g; s/[^a-zA-Z0-9_]/_/g')
    local log_file="$LOG_DIR/$(printf "%02d" $TEST_INDEX)_${safe_name}.log"

    banner "$test_name" "$cmd"

    if eval "$cmd" >"$log_file" 2>&1; then
        echo -e "${GREEN}[PASS]${NC} $test_name"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}[FAIL]${NC} $test_name"
        echo -e "${RED}  ┌─ Last 20 lines of $log_file:${NC}"
        tail -n 20 "$log_file" | while IFS= read -r line; do
            echo -e "${RED}  │ $line${NC}"
        done
        echo -e "${RED}  └────────────────────────────────────────────────────────${NC}"
        FAIL=$((FAIL + 1))
    fi

    TEST_INDEX=$((TEST_INDEX + 1))
}

# ── Header ─────────────────────────────────────────────────────────────────
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  Exhaustive Smoke Test Suite${NC}"
echo -e "${YELLOW}  $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${YELLOW}  Log directory: $LOG_DIR${NC}"
echo -e "${YELLOW}========================================${NC}"

# ===========================================================================
# MODULE IMPORT VERIFICATION (Refactored paths)
# ===========================================================================
echo -e "${CYAN}>> Phase 1: Verifying all refactored module imports...${NC}"

run_test \
    "Import_all_refactored_modules" \
    "python -c \"from src.models import MultiTaskUNet, GradNormBalancer; from src.metrics import dice_coefficient; from src.loader_tuning import resolve_batch_size, _initial_loader_tuning, _select_cache_size, _logical_cpu_count, _available_ram_gb; from src.checkpoints import save_checkpoint, load_checkpoint, save_training_state, load_training_state; from src.training import train_single_run, _run_epoch; from src.config import REPRO_DISABLE_CUDNN, REPRO_STRICT_BATCH_CHECKS, REPRO_ALLOW_BIG_CACHE, REPRO_ALLOW_UNC_WORKERS, REPRO_TORCH_COMPILE_BACKEND; print('All imports OK')\""

# ===========================================================================
# TRAINING SMOKE TESTS
# ===========================================================================
# --smoke-test forces 1 epoch, 2 batches max, no checkpoint saving.
# --compile is appended to all tests; main.py auto-disables it when
# GradNorm is active (use_gradnorm=True), so V2 runs are safe.

# ===========================================================================
# A. ENCODER COVERAGE: VGG16
# ===========================================================================
echo -e "${CYAN}>> Phase 2A: Encoder coverage — VGG16${NC}"

# A1. V1 (static loss) + VGG16
run_test \
    "Encoder_VGG16_V1_TCGA" \
    "python main.py --phase v1 --datasets tcga --encoders vgg16 --compile --smoke-test"

# A2. V2 (GradNorm) + VGG16
run_test \
    "Encoder_VGG16_V2_PanNuke" \
    "python main.py --phase v2 --datasets pannuke --encoders vgg16 --compile --smoke-test"

# ===========================================================================
# B. ENCODER COVERAGE: MobileNetV2
# ===========================================================================
echo -e "${CYAN}>> Phase 2B: Encoder coverage — MobileNetV2${NC}"

# B1. V1 (static loss) + MobileNetV2
run_test \
    "Encoder_MobileNetV2_V1_PANDA" \
    "python main.py --phase v1 --datasets panda --encoders mobilenet_v2 --compile --smoke-test"

# B2. V2 (GradNorm) + MobileNetV2
run_test \
    "Encoder_MobileNetV2_V2_SIIM" \
    "python main.py --phase v2 --datasets siim --encoders mobilenet_v2 --compile --smoke-test"

# ===========================================================================
# C. PHASE COVERAGE: V1 (Static Loss)
# ===========================================================================
echo -e "${CYAN}>> Phase 2C: Phase coverage — V1 (static loss weights)${NC}"

# C1. V1 + TCGA + VGG16 (already covered above, but explicit V1 label)
run_test \
    "Phase_V1_TCGA_VGG16_no_macenko" \
    "python main.py --phase v1 --datasets tcga --encoders vgg16 --no-macenko --disable-gradnorm --static-weights --compile --smoke-test"

# C2. V1 + PANDA + MobileNetV2
run_test \
    "Phase_V1_PANDA_MobileNetV2_no_macenko" \
    "python main.py --phase v1 --datasets panda --encoders mobilenet_v2 --no-macenko --disable-gradnorm --static-weights --compile --smoke-test"

# C3. V1 + Macenko ON (tests V1 path WITH preprocessing)
run_test \
    "Phase_V1_TCGA_MobileNetV2_with_macenko" \
    "python main.py --phase v1 --datasets tcga --encoders mobilenet_v2 --disable-gradnorm --static-weights --compile --smoke-test"

# ===========================================================================
# D. PHASE COVERAGE: V2 (GradNorm Dynamic Loss)
# ===========================================================================
echo -e "${CYAN}>> Phase 2D: Phase coverage — V2 (GradNorm dynamic loss)${NC}"

# D1. V2 + SIIM + MobileNetV2 (already covered above, but explicit V2 label)
run_test \
    "Phase_V2_SIIM_MobileNetV2" \
    "python main.py --phase v2 --datasets siim --encoders mobilenet_v2 --compile --smoke-test"

# D2. V2 + PanNuke + VGG16 (GradNorm + Macenko default ON)
run_test \
    "Phase_V2_PanNuke_VGG16" \
    "python main.py --phase v2 --datasets pannuke --encoders vgg16 --compile --smoke-test"

# D3. V2.1 control variant
run_test \
    "Phase_V2.1_PanNuke_MobileNetV2" \
    "python main.py --phase v2.1 --datasets pannuke --encoders mobilenet_v2 --compile --smoke-test"

# ===========================================================================
# E. ABLATION: --no-skip-connections
# ===========================================================================
echo -e "${CYAN}>> Phase 2E: Ablation — skip connections disabled${NC}"

# E1. V2 + TCGA + MobileNetV2 + no skip
run_test \
    "Ablation_NoSkip_V2_TCGA_MobileNetV2" \
    "python main.py --phase v2 --datasets tcga --encoders mobilenet_v2 --no-skip-connections --compile --smoke-test"

# E2. V2 + PANDA + MobileNetV2 + no skip
run_test \
    "Ablation_NoSkip_V2_PANDA_MobileNetV2" \
    "python main.py --phase v2 --datasets panda --encoders mobilenet_v2 --no-skip-connections --compile --smoke-test"

# ===========================================================================
# F. ABLATION: --no-macenko (Preprocessing OFF)
# ===========================================================================
echo -e "${CYAN}>> Phase 2F: Ablation — Macenko normalization disabled${NC}"

# F1. V2 + PanNuke + MobileNetV2 + no macenko
run_test \
    "Ablation_NoMacenko_V2_PanNuke_MobileNetV2" \
    "python main.py --phase v2 --datasets pannuke --encoders mobilenet_v2 --no-macenko --compile --smoke-test"

# F2. V2 + PANDA + MobileNetV2 + no macenko
run_test \
    "Ablation_NoMacenko_V2_PANDA_MobileNetV2" \
    "python main.py --phase v2 --datasets panda --encoders mobilenet_v2 --no-macenko --compile --smoke-test"

# ===========================================================================
# G. ABLATION: Loss weights override
# ===========================================================================
echo -e "${CYAN}>> Phase 2G: Ablation — custom loss weights${NC}"

# G1. Classification-heavy weights
run_test \
    "Ablation_Loss_PANDA_VGG16_cls_heavy" \
    "python main.py --phase v2 --datasets panda --encoders vgg16 --lambda-seg 1 --lambda-cls 10 --compile --smoke-test"

# G2. Segmentation-heavy weights
run_test \
    "Ablation_Loss_PANDA_VGG16_seg_heavy" \
    "python main.py --phase v2 --datasets panda --encoders vgg16 --lambda-seg 10 --lambda-cls 1 --compile --smoke-test"

# ===========================================================================
# H. ABLATION: GradNorm alpha override
# ===========================================================================
echo -e "${CYAN}>> Phase 2H: Ablation — GradNorm alpha override${NC}"

run_test \
    "Ablation_GradNormAlpha_PANDA_VGG16_alpha05" \
    "python main.py --phase v2 --datasets panda --encoders vgg16 --gradnorm-alpha 0.5 --compile --smoke-test"

# ===========================================================================
# SUMMARY
# ===========================================================================
TOTAL=$((PASS + FAIL))

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  SMOKE TEST SUMMARY${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "  Total : $TOTAL"
echo -e "  ${GREEN}Passed: $PASS${NC}"
echo -e "  ${RED}Failed: $FAIL${NC}"
echo -e "  Logs  : $LOG_DIR/${NC}"
echo -e "${BLUE}============================================================${NC}"

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}Some smoke tests failed. See $LOG_DIR/ for individual logs.${NC}"
    exit 1
fi

echo -e "${GREEN}All smoke tests passed.${NC}"
exit 0