#!/usr/bin/env bash
# Launch every run listed in a grid YAML (default: configs/grid.yaml).
# Usage:
#   bash scripts/run_grid.sh                           # all runs in configs/grid.yaml
#   bash scripts/run_grid.sh configs/grid.yaml must    # only priority=must
#   bash scripts/run_grid.sh configs/other-grid.yaml   # alternative grid file
#
# Server-only. Expects: training venv activated; `verl` importable; CUDA working.
# Each run launches via scripts/launch_run.sh so results land under results/<run_id>/.
# Sequential by design: an early schema failure must halt before wasting compute.
set -euo pipefail

GRID=${1:-configs/grid.yaml}
PRIORITY=${2:-all}  # all | must | should | stretch

cd "$(dirname "$0")/.."

if [[ ! -f "${GRID}" ]]; then
  echo "ERROR: grid file not found: ${GRID}" >&2
  exit 2
fi

CONFIGS=$(python -c "
import yaml, sys
grid = yaml.safe_load(open('${GRID}'))
want = '${PRIORITY}'
for r in grid.get('runs', []):
    if want == 'all' or r.get('priority') == want:
        print(r['config'])
")

if [[ -z "${CONFIGS}" ]]; then
  echo "ERROR: no runs matching priority='${PRIORITY}' in ${GRID}" >&2
  exit 3
fi

echo "=== grid: ${GRID} (priority=${PRIORITY}) ==="
echo "${CONFIGS}" | nl -ba

for cfg in ${CONFIGS}; do
  echo "---"
  echo "LAUNCHING: ${cfg}"
  if ! bash scripts/launch_run.sh "${cfg}"; then
    echo "FAILED: ${cfg}" >&2
    echo "Grid halted. Fix the config / verl schema error, then re-run with" >&2
    echo "  bash scripts/run_grid.sh ${GRID} ${PRIORITY}" >&2
    echo "already-completed runs will be re-launched; use a tighter grid file to skip." >&2
    exit 1
  fi
done

echo "=== grid complete ==="
