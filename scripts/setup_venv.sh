#!/usr/bin/env bash
# Laptop venv setup. Creates .venv/ in the project root using Python 3.12
# (safer PyTorch wheel coverage than 3.14), installs the project + CUDA
# PyTorch + inference extras.
#
# Idempotent: if .venv already exists, it's reused.
#
# Usage (Git Bash on Windows):
#     bash scripts/setup_venv.sh
#
# After it finishes:
#     source .venv/Scripts/activate       (Git Bash)
#     .venv\Scripts\Activate.ps1          (PowerShell)
#
# For the SERVER setup (verl + vLLM), use scripts/setup_env.sh instead.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

# ---- 1. find Python 3.12 ---------------------------------------------
PY=""
for cand in "py -3.12" "python3.12" "/c/Users/Chaitanya/AppData/Local/Programs/Python/Python312/python.exe"; do
    if $cand -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,12) else 1)" >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done

if [[ -z "$PY" ]]; then
    echo "ERROR: Python 3.12 not found."
    echo "Install from https://www.python.org/downloads/release/python-3127/"
    echo "(or whatever current 3.12.x is), then re-run this script."
    exit 2
fi

echo "Using: $($PY -c 'import sys; print(sys.executable)')"
echo "       $($PY --version 2>&1)"
echo

# ---- 2. create venv --------------------------------------------------
if [[ -d "$VENV_DIR" ]]; then
    echo "[1/6] .venv already exists; reusing it."
else
    echo "[1/6] Creating .venv with Python 3.12..."
    $PY -m venv "$VENV_DIR"
fi

# ---- 3. activate (via sourced path, since this is bash) --------------
# shellcheck disable=SC1091
source "$VENV_DIR/Scripts/activate"
echo "[2/6] Activated: $(which python)"
python --version

# ---- 4. upgrade pip + install project --------------------------------
echo
echo "[3/6] Upgrading pip..."
python -m pip install --upgrade pip -q

echo
echo "[4/6] Installing project + [dev] extras (editable)..."
python -m pip install -e "$PROJECT_ROOT[dev]"

# ---- 5. install CUDA torch -------------------------------------------
echo
echo "[5/6] Installing CUDA torch from $TORCH_INDEX ..."
# `-U` so an existing CPU torch inside the venv is replaced.
python -m pip install -U --index-url "$TORCH_INDEX" \
    torch torchvision torchaudio
TORCH_RC=$?

if [[ $TORCH_RC -ne 0 ]]; then
    echo
    echo "Torch install failed. Try CUDA 12.4 wheels instead:"
    echo "    python -m pip install -U --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio"
    exit 1
fi

# ---- 6. inference extras ---------------------------------------------
echo
echo "[6/6] Installing inference extras (accelerate, peft, sentence-transformers, bert-score)..."
python -m pip install -q accelerate peft sentence-transformers bert-score

# ---- 7. verify -------------------------------------------------------
echo
echo "=== Verification ==="
python -c "
import torch, sys
print(f'python:      {sys.version.split()[0]}')
print(f'torch:       {torch.__version__}')
print(f'cuda avail:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'cuda ver:    {torch.version.cuda}')
    print(f'device:      {torch.cuda.get_device_name(0)}')
    gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'VRAM:        {gb:.1f} GB')
else:
    print('WARNING: torch installed but CUDA not available in this venv.')
    print('Check: nvidia-smi works? torch is the +cu* wheel, not +cpu?')
    sys.exit(1)
"

echo
echo "Done. Next time you open a terminal in this project, run:"
echo "    source .venv/Scripts/activate"
echo
echo "Then try the local sanity check:"
echo "    python scripts/local_sanity.py --n 3 --device cuda --max-new-tokens 128"
