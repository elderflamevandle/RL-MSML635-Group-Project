#!/usr/bin/env bash
# One-shot environment setup for Google Colab (T4 GPU, 16 GB VRAM, Linux).
#
# This used to hand-build a Python 3.12 venv and pip-install torch/vLLM/verl
# one package at a time -- which is exactly how you get the "pyarrow wants
# numpy>=2 but vllm wants numpy<2" class of conflict, since pip resolves each
# install call mostly independently. `uv` resolves the whole dependency graph
# from one lockfile (uv.lock) in one shot, so those conflicts get caught (and
# fixed, once) before anything installs, not half-way through training.
#
# What it actually does, now delegated to scripts/setup_uv.sh:
#   1. install uv (it downloads its own Python 3.12 -- Colab's kernel is on
#      3.13, which verl's own pyproject.toml rejects)
#   2. `uv sync --extra train` against the committed uv.lock -> .venv
#   3. clone verl at the GRACE pin into .vendor/verl
#
# Usage (from a notebook cell):
#     !bash scripts/setup_colab.sh
#
# Idempotent. ~5-6 min first run (mostly the torch/vLLM wheel downloads),
# ~30s after uv's cache is warm.
set -euo pipefail
cd "$(dirname "$0")/.."

# Keep the venv on local disk, not Drive -- Drive's FUSE layer makes a
# site-packages tree with tens of thousands of small files painfully slow,
# and uv's hardlink cache doesn't work across the Drive mount anyway.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/content/venv}"
bash scripts/setup_uv.sh

# VERL_PY (read by scripts/launch_run.sh and the eval-driver notebook cell)
# should point at this interpreter, not a repo-relative .venv, since the
# venv lives outside the Drive-mounted repo on Colab.
echo "VERL_PY=${UV_PROJECT_ENVIRONMENT}/bin/python"
