#!/usr/bin/env bash
# Server-side env setup. Run on the GPU box.
# Pre-reqs: Python 3.12, CUDA 12.4 driver, GPU with >=16GB VRAM.
#
# For laptop/dev-only setup, use `scripts/install_cuda_torch.sh` instead.
# That script skips verl + vllm (which don't belong on a laptop).
set -euo pipefail

PY=${PY:-python3.12}

# GRACE paper's pinned verl commit (from https://github.com/YiboZhao624/Grace README).
VERL_COMMIT="${VERL_COMMIT:-4aa02fe1663d8048b9c204345b2abe5197870df3}"

$PY -m venv ~/rl-rag-venv
source ~/rl-rag-venv/bin/activate
pip install --upgrade pip

# Install CUDA-capable torch FIRST so later installs don't pull in the CPU wheel.
# cu121 wheels are compatible with CUDA 12.3/12.4 drivers.
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Project deps (uses pyproject.toml extras: [dev] + [train]).
pip install -e ".[dev,train]"

# verl at pinned commit.
pip install "git+https://github.com/volcengine/verl.git@${VERL_COMMIT}"

# Verify CUDA.
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available after setup'
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count())
print('device 0:', torch.cuda.get_device_name(0))
"

# Verify vllm.
python -c "import vllm; print('vllm', vllm.__version__)"

echo "Setup OK."
