#!/usr/bin/env bash
# One-shot environment setup for Google Colab (T4 GPU, 16 GB VRAM, Linux).
#
# Colab ships Python 3.11 by default; the project's pyproject.toml pins
# ">=3.12". We bypass that by installing dependencies directly with pip
# (no `pip install -e .`) and relying on scripts/*.py doing
# `sys.path.insert(0, repo_root)` for imports. All business logic in src/
# works on 3.11 fine.
#
# Usage (from a notebook cell):
#     !bash scripts/setup_colab.sh
#
# Idempotent: re-running it is safe.
set -uo pipefail

echo "[1/6] python + pip versions"
python --version
pip --version

echo "[2/6] CUDA-capable torch (cu121 wheels, work on T4 sm_75)"
pip install -q --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" || {
      echo "cu121 torch install failed; trying cu124"
      pip install -q --index-url https://download.pytorch.org/whl/cu124 \
          torch torchvision torchaudio
    }

echo "[3/6] HF + retrieval + eval deps"
pip install -q \
    "datasets>=2.20,<5" \
    "transformers>=4.44,<5" \
    "sentence-transformers>=3.0,<5" \
    "huggingface-hub<1.0" \
    "rank-bm25>=0.2.2" \
    "rouge-score>=0.1.2" \
    "bert-score>=0.3.13" \
    "numpy<2" \
    "pydantic>=2" \
    "pyyaml" \
    "requests" \
    "tenacity" \
    "accelerate" \
    "peft" \
    "wandb"

echo "[4/6] vLLM (rollout backend)"
# Pin a vLLM known to work with torch 2.5 / cu121 on T4.
pip install -q "vllm==0.6.6" || {
  echo "vllm==0.6.6 install failed; trying latest"
  pip install -q vllm
}

echo "[5/6] verl @ GRACE-pinned commit (editable clone — wheel builds drop subpackages like experimental/dataset)"
VERL_COMMIT="${VERL_COMMIT:-4aa02fe1663d8048b9c204345b2abe5197870df3}"
VERL_DIR="${VERL_DIR:-/tmp/verl_src}"
pip uninstall -y verl >/dev/null 2>&1 || true
if [ -d "$VERL_DIR/.git" ]; then
    git -C "$VERL_DIR" fetch --quiet origin
else
    rm -rf "$VERL_DIR"
    git clone --quiet https://github.com/volcengine/verl.git "$VERL_DIR"
fi
git -C "$VERL_DIR" checkout --quiet "$VERL_COMMIT"
pip install -q -e "$VERL_DIR"

echo "[6/6] verify"
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"device 0: {p.name}  compute capability: sm_{p.major}{p.minor}  VRAM: {p.total_memory/1024**3:.1f} GB")
try:
    import vllm
    print("vllm:", vllm.__version__)
except Exception as e:
    print("vllm import failed:", e)
try:
    import verl
    print("verl: importable")
except Exception as e:
    print("verl import failed:", e)
PY

echo "Setup OK."
