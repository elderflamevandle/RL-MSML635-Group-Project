#!/usr/bin/env bash
# Usage: launch_run.sh <config.yaml>
# Server-only — expects verl + CUDA env already set up.
set -euo pipefail
CFG=${1:?"usage: launch_run.sh <config.yaml>"}

# Expect the user to have activated their training venv first.
# source ~/rl-rag-venv/bin/activate  # team member runs this

cd "$(dirname "$0")/.."

RUN_ID=$(python -c "
import yaml
from src.training.run_naming import make_run_id
c = yaml.safe_load(open('${CFG}'))
print(make_run_id(c['_meta']['reward'], c['_meta']['optimizer'], c['_meta']['model'], c['_meta'].get('seed', 0)))
")
OUT="results/${RUN_ID}"
mkdir -p "${OUT}"
cp "${CFG}" "${OUT}/config.yaml"

echo "RUN_ID=${RUN_ID}"
echo "OUT=${OUT}"

python -m verl.trainer.main_ppo --config "${CFG}" \
  trainer.default_local_dir="${OUT}" \
  2>&1 | tee "${OUT}/train.log"
