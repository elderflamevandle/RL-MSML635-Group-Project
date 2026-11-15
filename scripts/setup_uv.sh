#!/usr/bin/env bash
# One-shot environment setup with uv. Works on Colab, Kaggle, or any Linux GPU box.
#
#   bash scripts/setup_uv.sh            # full training stack (torch + vLLM + verl deps)
#   bash scripts/setup_uv.sh --lite     # no vLLM/ray: enough for eval + the HF fallback trainer
#
# What it does:
#   1. installs uv if missing
#   2. `uv sync --frozen` -> uv downloads Python 3.12 itself and installs the
#      exact versions in uv.lock into an isolated venv. Nothing is installed
#      into the notebook kernel, so Colab's preinstalled numpy/pyarrow can't clash.
#   3. clones verl at the GRACE pin into .vendor/verl (used via PYTHONPATH)
#   4. import-checks the stack
#
# Env vars:
#   UV_PROJECT_ENVIRONMENT  where the venv lives (default: .venv; the notebooks
#                           point it at local disk because Drive is slow)
#   VERL_DIR                where verl is cloned (default: .vendor/verl)
set -euo pipefail
cd "$(dirname "$0")/.."

LITE=0
[ "${1:-}" = "--lite" ] && LITE=1

echo "[1/4] uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "[2/4] sync locked environment"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
if [ "$LITE" = "1" ]; then
    uv sync --frozen
else
    uv sync --frozen --extra train
fi
VENV="${UV_PROJECT_ENVIRONMENT:-$(pwd)/.venv}"
PY="${VENV}/bin/python"

VERL_COMMIT="${VERL_COMMIT:-4aa02fe1663d8048b9c204345b2abe5197870df3}"
VERL_DIR="${VERL_DIR:-$(pwd)/.vendor/verl}"
if [ "$LITE" = "0" ]; then
    echo "[3/4] verl @ ${VERL_COMMIT:0:7} -> ${VERL_DIR}"
    if [ ! -d "${VERL_DIR}/.git" ]; then
        rm -rf "${VERL_DIR}"
        mkdir -p "$(dirname "${VERL_DIR}")"
        git clone --quiet https://github.com/volcengine/verl.git "${VERL_DIR}"
    fi
    git -C "${VERL_DIR}" fetch --quiet origin || true
    git -C "${VERL_DIR}" checkout --quiet "${VERL_COMMIT}"
else
    echo "[3/4] verl skipped (--lite)"
fi

echo "[4/4] verify"
LITE="$LITE" PYTHONPATH="$(pwd):${VERL_DIR}" "$PY" - <<'PY'
import os, sys
import numpy, pyarrow, pyarrow.parquet, datasets, torch, transformers, peft
print("python      :", sys.version.split()[0])
print("torch       :", torch.__version__, "| cuda", torch.version.cuda, "| available:", torch.cuda.is_available())
print("numpy       :", numpy.__version__)
print("pyarrow     :", pyarrow.__version__)
print("datasets    :", datasets.__version__)
print("transformers:", transformers.__version__)
print("peft        :", peft.__version__)
from transformers import AutoConfig
assert "qwen3" in transformers.models.auto.configuration_auto.CONFIG_MAPPING_NAMES, "transformers too old for Qwen3"
if os.environ["LITE"] == "0":
    import ray, tensordict, vllm
    from vllm.model_executor.models.registry import ModelRegistry
    assert "Qwen3ForCausalLM" in ModelRegistry.get_supported_archs(), "vllm too old for Qwen3"
    import verl.trainer.main_ppo  # noqa: F401
    from verl.utils.vllm_utils import is_version_ge
    assert is_version_ge(pkg="vllm", minver="0.7.3")
    print("vllm        :", vllm.__version__, "| ray", ray.__version__, "| tensordict", tensordict.__version__)
    print("verl        : OK")
PY
echo "Setup OK. Interpreter: ${PY}"
