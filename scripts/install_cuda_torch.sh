#!/usr/bin/env bash
# Install a CUDA-capable PyTorch build on a laptop.
#
# Current detected state (as of 2026-10-14):
#   - You have a CPU-only torch (`torch==2.11.0+cpu`).
#   - Your GPU is an RTX 4050 Laptop (6GB) with CUDA 12.3 driver.
#   - PyTorch's cu121 wheels work with CUDA 12.3 (forward compatible).
#   - On Windows Python installs, bare `pip` is usually NOT on PATH;
#     this script uses `python -m pip` which always works.
#
# Usage (Git Bash on Windows):
#     bash scripts/install_cuda_torch.sh
#
# Usage (PowerShell):
#     python -m pip uninstall -y torch torchvision torchaudio
#     python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
#     python -m pip install accelerate peft sentence-transformers bert-score
#
# Expected disk + time: ~2.5 GB download, 5-10 min.

set -uo pipefail  # NOT -e: we tolerate uninstall of absent packages

# ---- find a working python interpreter ------------------------------------
PY=""
for candidate in python python3 py; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >/dev/null 2>&1; then
            PY="$candidate"
            break
        fi
    fi
done

if [[ -z "$PY" ]]; then
    echo "ERROR: no Python >= 3.10 found on PATH."
    echo "Tried: python, python3, py"
    exit 2
fi

echo "Using interpreter: $($PY -c 'import sys; print(sys.executable)')"
echo "Python version:    $($PY --version 2>&1)"
echo

PIP="$PY -m pip"

# ---- step 1: uninstall CPU torch ------------------------------------------
echo "[1/4] Uninstalling CPU-only torch (if present)..."
$PIP uninstall -y torch torchvision torchaudio 2>/dev/null || true

# ---- step 2: install CUDA torch -------------------------------------------
echo
echo "[2/4] Installing CUDA 12.1 torch wheels..."
$PIP install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

INSTALL_RC=$?
if [[ $INSTALL_RC -ne 0 ]]; then
    echo
    echo "CUDA torch install failed (exit $INSTALL_RC)."
    echo "Common causes:"
    echo "  - Python 3.14 is very new; cu121 wheels may lag."
    echo "    Try CUDA 12.4 wheels instead:"
    echo "      $PIP install --index-url https://download.pytorch.org/whl/cu124 torch"
    echo "  - Or use nightly build (bleeding edge, Python 3.14+):"
    echo "      $PIP install --pre --index-url https://download.pytorch.org/whl/nightly/cu124 torch"
    echo "  - Or downgrade to Python 3.12 (most reliable)."
    exit 1
fi

# ---- step 3: install inference-side extras --------------------------------
echo
echo "[3/4] Installing inference extras (accelerate, peft, sentence-transformers, bert-score)..."
$PIP install accelerate peft sentence-transformers bert-score

# ---- step 4: verify -------------------------------------------------------
echo
echo "[4/4] Verifying CUDA..."
$PY -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device name:', torch.cuda.get_device_name(0))
    print('total VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
else:
    print()
    print('WARNING: torch installed but cuda not available.')
    print('Check: nvidia-smi works? torch wheel is cu* not +cpu?')
"

echo
echo "If 'cuda available: True' above, try:"
echo "    $PY scripts/local_sanity.py --n 3 --device cuda --max-new-tokens 128"
