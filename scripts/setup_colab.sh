#!/usr/bin/env bash
# One-shot environment setup for Google Colab (T4 GPU, 16 GB VRAM, Linux).
#
# Colab's default Python is 3.13; verl's own pyproject.toml pins "<3.13,>=3.10"
# and refuses to install. Fix: create a Python 3.12 venv at /content/venv312
# and install the full verl stack THERE. Jupyter's Python 3.13 keeps running
# the orchestration cells (data download, baseline eval, result parsing)
# with a minimal HF stack.
#
# Usage (from a notebook cell):
#     !bash scripts/setup_colab.sh
#
# Idempotent. ~10 min first run, ~1 min after wheels are cached.
set -uo pipefail

echo "[1/5] Python 3.12 (apt may prompt for sudo privilege)"
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12 python3.12-dev python3.12-venv

echo "[2/5] minimal HF stack in Jupyter's Python 3.13 (for data + baseline eval)"
# transformers>=4.51 is required for Qwen3ForCausalLM; older pins silently
# install a version that doesn't know the Qwen3 architecture.
pip install -q "numpy<2" \
    "datasets>=2.20,<5" "transformers>=4.51,<5" \
    "rank-bm25>=0.2.2" "rouge-score>=0.1.2" "bert-score>=0.3.13" \
    "huggingface-hub<1.0" "pydantic>=2" "pyyaml" "requests" "tenacity"

echo "[3/5] create venv312"
python3.12 -m venv /content/venv312 --upgrade-deps
PIP="/content/venv312/bin/pip"

echo "[4/5] torch cu121 + HF + vllm 0.6.6 into venv312"
$PIP install -q --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"
$PIP install -q \
    "numpy<2" "datasets>=2.20,<5" "transformers>=4.51,<5" \
    "sentence-transformers>=3.0,<5" "huggingface-hub<1.0" \
    "rank-bm25>=0.2.2" "rouge-score>=0.1.2" "bert-score>=0.3.13" \
    "pydantic>=2" "pyyaml" "requests" "tenacity" \
    "accelerate" "peft" "wandb" "hydra-core>=1.3"
$PIP install -q "vllm==0.6.6"

echo "[5a/5] verl @ GRACE pin, editable (wheel drops subpackages like experimental/dataset)"
VERL_COMMIT="${VERL_COMMIT:-4aa02fe1663d8048b9c204345b2abe5197870df3}"
VERL_DIR="${VERL_DIR:-/tmp/verl_src}"
$PIP uninstall -y verl >/dev/null 2>&1 || true
if [ -d "$VERL_DIR/.git" ]; then
    git -C "$VERL_DIR" fetch --quiet origin
else
    rm -rf "$VERL_DIR"
    git clone --quiet https://github.com/volcengine/verl.git "$VERL_DIR"
fi
git -C "$VERL_DIR" checkout --quiet "$VERL_COMMIT"
$PIP install -q -e "$VERL_DIR"

echo "[5b/5] verify"
/content/venv312/bin/python - <<'PY'
import sys, torch, vllm, verl
from verl.utils.vllm_utils import is_version_ge
print("python:", sys.version.split()[0])
print("torch :", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("vllm  :", vllm.__version__)
print("verl  : OK")
PY

echo "Setup OK."
