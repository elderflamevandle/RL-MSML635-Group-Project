#!/usr/bin/env bash
# Usage: launch_run.sh <config.yaml>
# Runs verl's PPO/GRPO trainer. On Colab, verl lives in /content/venv312 (its
# own pyproject pins Python <3.13 which Colab's kernel does not satisfy).
# Export VERL_PY to override the Python used; defaults to the venv if present,
# else system python.
set -euo pipefail
CFG=${1:?"usage: launch_run.sh <config.yaml>"}

cd "$(dirname "$0")/.."

# Pick the Python that has verl installed. Honor explicit override first.
if [ -n "${VERL_PY:-}" ] && [ -x "${VERL_PY}" ]; then
    PY="${VERL_PY}"
elif [ -x "/content/venv312/bin/python" ]; then
    PY="/content/venv312/bin/python"
else
    PY="python"
fi

# Our YAMLs carry a custom `_meta` block (reward/optimizer/model/seed) that
# verl's struct-mode Hydra schema does NOT allow. Extract it here for the
# run_id, then write a stripped copy for verl to consume.
RUN_ID=$($PY -c "
import yaml, sys
sys.path.insert(0, '.')
from src.training.run_naming import make_run_id
c = yaml.safe_load(open('${CFG}'))
m = c['_meta']
print(make_run_id(m['reward'], m['optimizer'], m['model'], m.get('seed', 0)))
")
OUT="results/${RUN_ID}"
mkdir -p "${OUT}"
cp "${CFG}" "${OUT}/config.yaml"

$PY - <<PY_EOF
import yaml
c = yaml.safe_load(open("${CFG}"))
c.pop("_meta", None)
open("${OUT}/config_verl.yaml", "w").write(yaml.safe_dump(c, sort_keys=False))
PY_EOF

echo "RUN_ID=${RUN_ID}"
echo "OUT=${OUT}"
echo "PY=${PY}"

# verl uses Hydra. Our config extends verl's ppo_trainer base via `defaults:`.
# --config-dir adds verl's config dir to Hydra's search path so `ppo_trainer`
# resolves. --config-path points at where our stripped YAML lives.
OUT_ABS=$(realpath "${OUT}")
VERL_CFG_DIR="${VERL_CFG_DIR:-/tmp/verl_src/verl/trainer/config}"

if [ ! -d "${VERL_CFG_DIR}" ]; then
    echo "ERROR: verl config dir not found at ${VERL_CFG_DIR}" >&2
    echo "Reinstall verl via scripts/setup_colab.sh or export VERL_CFG_DIR=..." >&2
    exit 2
fi

$PY -m verl.trainer.main_ppo \
  --config-path "${OUT_ABS}" \
  --config-name "config_verl" \
  --config-dir "${VERL_CFG_DIR}" \
  ++trainer.default_local_dir="${OUT}" \
  2>&1 | tee "${OUT}/train.log"
