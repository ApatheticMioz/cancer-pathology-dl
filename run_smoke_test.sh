#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Exhaustive Smoke Test Suite (Expanded 10-Test Matrix)
# ---------------------------------------------------------------------------
# Validates every unique Phase/Encoder/Ablation/Flag combination before
# committing to the 26-run concurrent orchestrator for Q1 journal submission.
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
echo -e "${YELLOW}  Exhaustive Smoke Test Suite (10 tests)${NC}"
echo -e "${YELLOW}  $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${YELLOW}  Log directory: $LOG_DIR${NC}"
echo -e "${YELLOW}========================================${NC}"

# ===========================================================================
# TEST MATRIX (10 configurations)
# ===========================================================================
# Note: --compile is appended to all tests.  main.py automatically disables
# it when GradNorm is active (use_gradnorm=True), so V2 runs are safe.

# 1. Baseline V1 | TCGA | VGG16
run_test \
    "Baseline_V1_TCGA_VGG16" \
    "python main.py --phase v1 --datasets tcga --encoders vgg16 --compile --smoke-test"

# 2. Baseline V1 | PANDA | MobileNetV2
run_test \
    "Baseline_V1_PANDA_MobileNetV2" \
    "python main.py --phase v1 --datasets panda --encoders mobilenet_v2 --compile --smoke-test"

# 3. Enhanced V2 | SIIM | MobileNetV2 (Verify standard V2)
run_test \
    "Enhanced_V2_SIIM_MobileNetV2" \
    "python main.py --phase v2 --datasets siim --encoders mobilenet_v2 --compile --smoke-test"

# 4. Enhanced V2 | PanNuke | VGG16 (Verify PanNuke with GradNorm+Macenko)
run_test \
    "Enhanced_V2_PanNuke_VGG16" \
    "python main.py --phase v2 --datasets pannuke --encoders vgg16 --compile --smoke-test"

# 5. Control V2.1 | PanNuke | MobileNetV2
run_test \
    "Control_V2.1_PanNuke_MobileNetV2" \
    "python main.py --phase v2.1 --datasets pannuke --encoders mobilenet_v2 --compile --smoke-test"

# 6. Arch Ablation | TCGA | MobileNetV2 | --no-skip-connections
run_test \
    "Arch_Ablation_TCGA_MobileNetV2_no_skip" \
    "python main.py --phase v2 --datasets tcga --encoders mobilenet_v2 --no-skip-connections --compile --smoke-test"

# 7. Arch Ablation | PANDA | MobileNetV2 | --no-skip-connections
run_test \
    "Arch_Ablation_PANDA_MobileNetV2_no_skip" \
    "python main.py --phase v2 --datasets panda --encoders mobilenet_v2 --no-skip-connections --compile --smoke-test"

# 8. Loss Ablation | PANDA | VGG16 | --lambda-seg 1 --lambda-cls 10
run_test \
    "Loss_Ablation_PANDA_VGG16_cls_heavy" \
    "python main.py --phase v2 --datasets panda --encoders vgg16 --lambda-seg 1 --lambda-cls 10 --compile --smoke-test"

# 9. GradNorm Ablation | PANDA | VGG16 | --gradnorm-alpha 0.5
run_test \
    "GradNorm_Ablation_PANDA_VGG16_alpha05" \
    "python main.py --phase v2 --datasets panda --encoders vgg16 --gradnorm-alpha 0.5 --compile --smoke-test"

# 10. Domain Ablation | PanNuke | MobileNetV2 | --no-macenko
run_test \
    "Domain_Ablation_PanNuke_MobileNetV2_no_macenko" \
    "python main.py --phase v2 --datasets pannuke --encoders mobilenet_v2 --no-macenko --compile --smoke-test"

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