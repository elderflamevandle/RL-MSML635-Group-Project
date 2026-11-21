#!/usr/bin/env bash
# Usage: launch_run.sh <config.yaml> [extra hydra overrides...]
#
# Runs verl's PPO/GRPO trainer inside the uv environment.
#   - verl is a plain source checkout (scripts/setup_uv.sh clones it at the
#     GRACE pin); we put it on PYTHONPATH instead of pip-installing it.
#   - Our YAMLs carry a `_meta` block verl's strict schema rejects; it is
#     stripped into results/<run_id>/config_verl.yaml.
#   - RESUME=1 (default) reuses the newest run dir of the same
#     reward/optimizer/model/seed that already has checkpoints, so verl's
#     `resume_mode: auto` actually finds them. RESUME=0 forces a fresh run.
set -euo pipefail
CFG=${1:?"usage: launch_run.sh <config.yaml> [hydra overrides]"}
shift || true

cd "$(dirname "$0")/.."
REPO=$(pwd)

VERL_DIR="${VERL_DIR:-${REPO}/.vendor/verl}"
VERL_CFG_DIR="${VERL_DIR}/verl/trainer/config"
if [ ! -d "${VERL_CFG_DIR}" ]; then
    echo "ERROR: verl checkout not found at ${VERL_DIR}" >&2
    echo "Run: bash scripts/setup_uv.sh   (or export VERL_DIR=...)" >&2
    exit 2
fi
export PYTHONPATH="${REPO}:${VERL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

# Use the uv environment's interpreter DIRECTLY, not `uv run`: when Ray sees
# it was started under `uv run` it re-launches every worker through `uv run`
# too, which re-resolves the env without the [train] extra and the workers
# die with "No module named 'ray'".
VENV="${UV_PROJECT_ENVIRONMENT:-${REPO}/.venv}"
PY="${PY:-${VENV}/bin/python}"
if [ ! -x "${PY}" ]; then
    echo "ERROR: no interpreter at ${PY}. Run: bash scripts/setup_uv.sh" >&2
    exit 2
fi
RUN=("${PY}")
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

PREFIX=$("${RUN[@]}" -c "
import yaml
m = yaml.safe_load(open('${CFG}'))['_meta']
print(f\"{m['reward']}_{m['optimizer']}_{m['model']}_s{m.get('seed', 0)}\")
")

RUN_ID="${RUN_ID:-}"
if [ -z "${RUN_ID}" ] && [ "${RESUME:-1}" = "1" ]; then
    for d in $(ls -1d results/${PREFIX}_* 2>/dev/null | sort -r); do
        if ls -d "${d}"/checkpoints/global_step_* >/dev/null 2>&1; then
            RUN_ID=$(basename "${d}")
            echo "Resuming existing run: ${RUN_ID}"
            break
        fi
    done
fi
[ -n "${RUN_ID}" ] || RUN_ID="${PREFIX}_$(date +%Y%m%d-%H%M%S)"

OUT="results/${RUN_ID}"
mkdir -p "${OUT}"
cp "${CFG}" "${OUT}/config.yaml"
"${RUN[@]}" -c "
import yaml
c = yaml.safe_load(open('${CFG}'))
c.pop('_meta', None)
open('${OUT}/config_verl.yaml', 'w').write(yaml.safe_dump(c, sort_keys=False))
"

echo "RUN_ID=${RUN_ID}"
echo "OUT=${OUT}"

OUT_ABS=$(realpath "${OUT}")
"${RUN[@]}" -m verl.trainer.main_ppo \
  --config-path "${OUT_ABS}" \
  --config-name "config_verl" \
  --config-dir "${VERL_CFG_DIR}" \
  "++trainer.default_local_dir=${OUT_ABS}/checkpoints" \
  "$@" \
  2>&1 | tee -a "${OUT}/train.log"
