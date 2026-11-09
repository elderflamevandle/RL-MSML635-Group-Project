#!/usr/bin/env bash
# Usage: launch_run.sh <config.yaml>
# Server-only — expects verl + CUDA env already set up.
set -euo pipefail
CFG=${1:?"usage: launch_run.sh <config.yaml>"}

cd "$(dirname "$0")/.."

# Our YAMLs carry a custom `_meta` block (reward/optimizer/model/seed) that
# verl's struct-mode Hydra schema does NOT allow. Extract it here for the
# run_id, then write a stripped copy for verl to consume.
RUN_ID=$(python -c "
import yaml
from src.training.run_naming import make_run_id
c = yaml.safe_load(open('${CFG}'))
m = c['_meta']
print(make_run_id(m['reward'], m['optimizer'], m['model'], m.get('seed', 0)))
")
OUT="results/${RUN_ID}"
mkdir -p "${OUT}"
cp "${CFG}" "${OUT}/config.yaml"

python - <<PY
import yaml
c = yaml.safe_load(open("${CFG}"))
c.pop("_meta", None)
open("${OUT}/config_verl.yaml", "w").write(yaml.safe_dump(c, sort_keys=False))
PY

echo "RUN_ID=${RUN_ID}"
echo "OUT=${OUT}"

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

python -m verl.trainer.main_ppo \
  --config-path "${OUT_ABS}" \
  --config-name "config_verl" \
  --config-dir "${VERL_CFG_DIR}" \
  ++trainer.default_local_dir="${OUT}" \
  2>&1 | tee "${OUT}/train.log"
