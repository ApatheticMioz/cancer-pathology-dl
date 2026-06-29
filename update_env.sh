#!/usr/bin/env bash
# update_env.sh — Environment verification and conditional PyTorch upgrade.
#
# Activates the existing virtual environment (venv) in the project root,
# checks the installed PyTorch version, and upgrades only if it is older
# than 2.0.  PyTorch >= 2.0 is required for torch.compile support.
#
# Usage:
#   bash update_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

# ---------------------------------------------------------------------------
# Activate the existing venv
# ---------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "[ERROR] Virtual environment directory not found at: $VENV_DIR"
    echo "        Create one first:  python3 -m venv venv"
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[INFO] Virtual environment activated: $VENV_DIR"
echo "[INFO] Python : $(python --version 2>&1)"
echo "[INFO] Pip    : $(pip --version 2>&1)"

# ---------------------------------------------------------------------------
# Check PyTorch version
# ---------------------------------------------------------------------------
PYTORCH_VERSION=$(python -c "
import torch
parts = torch.__version__.split('+')[0].split('.')
major = int(parts[0])
minor = int(parts[1]) if len(parts) > 1 else 0
print(f'{major}.{minor}')
" 2>/dev/null || echo "0.0")

MAJOR=$(echo "$PYTORCH_VERSION" | cut -d. -f1)
MINOR=$(echo "$PYTORCH_VERSION" | cut -d. -f2)

echo "[INFO] Current PyTorch version: ${PYTORCH_VERSION} (full: $(python -c 'import torch; print(torch.__version__)'))"

if [ "$MAJOR" -gt 2 ] || { [ "$MAJOR" -eq 2 ] && [ "$MINOR" -ge 0 ]; }; then
    echo "[INFO] PyTorch >= 2.0 detected. No upgrade needed."
    echo "[INFO] torch.compile is available."
    exit 0
fi

# ---------------------------------------------------------------------------
# Upgrade PyTorch (version < 2.0)
# ---------------------------------------------------------------------------
echo "[WARN] PyTorch ${PYTORCH_VERSION} is older than 2.0. Upgrading..."

# Detect CUDA capability and pick the right index URL
CUDA_MAJOR=$(python -c "
import torch
if torch.cuda.is_available():
    prop = torch.cuda.get_device_properties(0)
    print(f'{prop.major}.{prop.minor}')
else:
    print('0.0')
" 2>/dev/null || echo "0.0")

CUDA_MAJOR_NUM=$(echo "$CUDA_MAJOR" | cut -d. -f1)

if [ "$CUDA_MAJOR_NUM" -ge 12 ]; then
    echo "[INFO] Installing PyTorch with CUDA 12.1+ support"
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [ "$CUDA_MAJOR_NUM" -ge 11 ]; then
    echo "[INFO] Installing PyTorch with CUDA 11.8 support"
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "[INFO] No CUDA detected or CUDA < 11. Installing CPU PyTorch"
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "[INFO] Upgrade complete. New PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "[INFO] Done."