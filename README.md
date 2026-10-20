# RL-Assisted RAG — Reward Design

MSML635 Group 5. Reward design for RL-assisted RAG, extending GRACE with cost-sensitive (R1) and hard-gated (R2) reward variants, plus optimizer comparison (GRPO vs DAPO) on HotpotQA with cross-domain stretch on QASPER.

- Spec: `docs/superpowers/specs/2026-10-11-rl-rag-reward-design-plan.md`
- Plan: `docs/superpowers/plans/2026-10-11-rl-rag-reward-design.md`
- **Live status:** `docs/STATUS.md` (what's done, what's left, what's possible where)
- **Target reproduction:** GRACE paper (<https://arxiv.org/abs/2601.04525>, repo <https://github.com/YiboZhao624/Grace>) — verl pinned at commit `4aa02fe1663d8048b9c204345b2abe5197870df3`.

---

## TL;DR — where we are

- ✅ Weeks 0-2 scaffolding complete and verified (37 → 93 tests passing locally)
- ✅ Week 3 Task 3.2 code done — R1 + R2 rewards + reward-hacking audit utilities
- ✅ Week 3 Task 3.1 code skeleton done — Groq judge with on-disk cache
- ✅ Week 4 Tasks 4.2, 4.3-code, 4.4-stretch done — aggregate, audit runner, QASPER loader
- ✅ `scripts/setup_env.sh` has the pinned verl commit filled in
- ✅ Local pipeline smoke-tested end-to-end on RTX 4050 (`scripts/local_sanity.py`)
- ✅ Report scaffolding + log templates pre-written

**Remaining critical path (requires humans + GPU server):**
Task 0.1 (team logistics, 30 min) → Task 1.1 (walking skeleton on server) → Task 2.1 server retrieval build → Task 2.3 server first R0+GRPO run (Week 2 Gate).

Detailed state in `docs/STATUS.md`. Full plan in `docs/superpowers/plans/`.

---

## Repository layout

```
src/
  common/      Shared types (Sample, ParsedResponse) + parse_response()
  data/        HotpotQA/QASPER loaders, retrievers (BM25 + Qwen3 dense), cache, balanced split
  prompts/     Single source of truth for prompt format (RAG_SYSTEM + build_prompt)
  rewards/     R0 (GRACE baseline), R1 (cost-sensitive), R2 (hard-gated), audit utilities
  eval/        evaluate.py CLI, metrics (EM/TNR/balanced), aggregate, Groq judge
  training/    run_naming (<reward>_<optimizer>_<model>_s<seed>_<ts>)
  experiments/ (empty — Week 4+)
configs/
  skeleton.yaml       — Week 1 walking skeleton (1 step, laptop sanity)
  r0_grpo.yaml        — Week 2 full-size GRPO baseline
scripts/
  download_hotpot_tiny.py  — 50 samples, laptop-safe
  download_hotpot_full.py  — 2000 train + 500 val
  build_retrieval.py       — BM25 + Qwen3 dense; cache; balanced split
  local_sanity.py          — base-model inference smoke test, laptop
  launch_run.sh            — server-only verl launcher
  setup_env.sh             — server-only env install (verl pinned at 4aa02fe)
  make_comparison_table.py — writes docs/results-table.{md,csv}
  run_audits.py            — writes results/<run>/audit.json
tests/            — 93 fast tests + 1 slow (dense retriever) passing
docs/
  STATUS.md             — current session snapshot (read first)
  team.md               — Task 0.1 fillable skeleton
  rewards.md            — reward design reference
  results-convention.md — results/<run_id>/ layout
  log/                  — TEMPLATE-skeleton-run.md, TEMPLATE-week2-gate.md
  report/               — REPORT.md + section stubs (reward-design/training/results/failure-cases)
  figures/              — Week 5 fills
data/
  raw/        HotpotQA jsonl (gitignored; regenerate with download_hotpot_*)
  cache/      bm25_{train,val}.jsonl + dense_{train,val}.jsonl + retrieval/*.json
results/      Each run: config.yaml, train.log, metrics.json, predictions.jsonl, audit.json
```

---

## Setup — laptop (Windows / macOS / Linux dev)

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -e ".[dev]"
pytest -q
```

**Expected:** `93 passed, 1 deselected` (slow test is the dense retriever's full model download — safe to deselect on laptops that already ran it).

To include the slow test:
```bash
pytest -q
```

### Known env knobs
- `pyproject.toml` pins `transformers<5`, `datasets<5`, `sentence-transformers<5` because torch 2.5 CUDA 12.1 wheels don't support `torch.accelerator` (added in torch 2.6). If you upgrade torch, you can relax all four.

---

## Setup — GPU server (Linux + CUDA 12.4 + ≥16 GB VRAM recommended)

```bash
bash scripts/setup_env.sh
```

Installs:
- `~/rl-rag-venv` with `torch` cu121, `transformers`, `vllm`, `peft`, `accelerate`, `wandb`
- `verl` pinned at `4aa02fe1663d8048b9c204345b2abe5197870df3` (GRACE's reference commit)

Expected final output: `Setup OK.` with torch CUDA available and vllm version printed.

---

## How to run — the critical path

### 1. Walking skeleton (Task 1.1, team on server)
```bash
bash scripts/launch_run.sh configs/skeleton.yaml
```
Expect failures. Verl's YAML field names drift between commits; read errors, patch `configs/skeleton.yaml`, retry. Record the final working config in `docs/log/<date>-skeleton-run.md` (copy from `docs/log/TEMPLATE-skeleton-run.md`).

### 2. Full retrieval build (Task 2.1 server portion)
```bash
python scripts/download_hotpot_full.py          # 2000 train + 500 val
PYTHONPATH=. python scripts/build_retrieval.py  # BM25 + Qwen3 dense, cached, balanced
```
Writes `data/cache/{bm25,dense}_{train,val}.jsonl` + `data/cache/retrieval/*.json` cache.

Flags on constrained GPUs:
- `--dense-device cpu` — force CPU embedding (slow but no VRAM)
- `--dense-batch 4` — small batch for low-VRAM GPUs
- `--no-dense` — BM25 only

Watch the printed `ans:unans` ratios. The script asserts `0.1 < ans_rate < 0.95` per split; if it fires, debug the gold-passage string normalization in `_hotpot_row_to_sample`.

### 3. First R0+GRPO run (Task 2.3 — Week 2 Gate)
```bash
bash scripts/launch_run.sh configs/r0_grpo.yaml                                  # 2-6h GPU
python -m src.eval.evaluate --checkpoint results/<run>/last \
    --split data/cache/dense_val.jsonl --out results/<run>/metrics.json
```
**Gate:** `balanced_accuracy ∈ [0.71, 0.91]` (within 10 points of GRACE's HotpotQA 0.81). If not, HALT; team pair-debugs before Week 3. Record in `docs/log/<date>-week2-gate.md` (copy from template).

### 4. Groq Judge validation (Task 3.1)
```bash
export GROQ_API_KEY=<key from password manager>
# Build 50-sample validation set:
python scripts/make_lj_validation.py                 # (Week 3 — to be written)
# Hand-label the 50 (humans).
python scripts/validate_lj.py                        # Must print agreement ≥ 85%.
```

### 5. Grid runs + audits (Tasks 4.1 + 4.3)
```bash
bash scripts/run_grid.sh must                        # (Week 3 — to be written)
python -m src.eval.evaluate --checkpoint ... --out ...   # per run
python scripts/run_audits.py                             # writes audit.json per run
python scripts/make_comparison_table.py                  # writes docs/results-table.{md,csv}
```

### 6. Report assembly (Task 5.4)
Fill in `docs/report/REPORT.md` + section stubs (`reward-design.md`, `training.md`, `results.md`, `failure-cases.md`). Figures to `docs/figures/`.

---

## Local sanity check (any laptop with CUDA)

Prove the full pipeline (data → prompt → generate → parse → score) works end-to-end against the base Qwen3-0.6B model:

```bash
python scripts/local_sanity.py --n 3      # fast: 3 samples, 37s each on RTX 4050
python scripts/local_sanity.py --n 20     # fuller baseline
python scripts/local_sanity.py --device cpu --n 3   # if no CUDA
```

Output lands in `results/local_sanity/<timestamp>/`. Expect low EM (untrained base model) — pipeline health is confirmed by exit code 0.

---

## Tests

```bash
pytest -q -m "not slow"   # 93 fast tests (~6s)
pytest -q                 # +1 slow dense-retriever test (~1 min, model download first run)
pytest tests/test_r1.py -v  # single file
```

TDD convention: every new reward/metric/loader gets failing-first tests. See `src/rewards/r0_grace.py` + `tests/test_r0.py` as the canonical example.

---

## Where to go next

- **If you've never seen this project:** read `docs/STATUS.md`, then `HANDOFF.md` for historical context, then this README.
- **If you're ready to execute:** `docs/team.md` → fill it. Then push to a remote, send clone URL to teammates, schedule the walking-skeleton session.
- **If you want to extend a reward:** read `docs/rewards.md`, then `src/rewards/r0_grace.py`, mimic the (config dataclass, pure function, `(reward, parts)` return) pattern.
- **If you want to extend the eval:** `src/eval/evaluate.py` runs the model; `src/eval/metrics.py` + `src/eval/judge.py` compute per-metric scores; `src/eval/aggregate.py` collects across runs.

---

## Troubleshooting cheat sheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `AttributeError: module 'torch' has no attribute 'accelerator'` | `transformers 5.x` installed over `torch 2.5` | `pip install "transformers>=4.44,<5"` (already pinned in pyproject) |
| `ImportError: cannot import name 'BucketNotFoundError'` | `datasets 5.x` over `huggingface_hub 0.x` | `pip install "datasets>=2.20,<5"` (already pinned in pyproject) |
| `Feature type 'List' not found` | `datasets < 4` loading newer dataset metadata | `pip install "datasets>=4,<5"` |
| `ModuleNotFoundError: No module named 'src'` running a script | script missing `sys.path.insert` | run as `PYTHONPATH=. python scripts/<x>.py` |
| `CUDA out of memory` during dense retrieval | Qwen3-Embedding batch too big for VRAM | `python scripts/build_retrieval.py --dense-batch 4` or `--dense-device cpu` |
| Walking skeleton rejects config field | verl API drift from GRACE's pinned commit | Read error, patch YAML, iterate; record in `docs/log/<date>-skeleton-run.md` |
| Week 2 Gate `balanced_accuracy` out of [0.71, 0.91] | Prompt drift, wrong LoRA rank, LR too high, path collapse | See `docs/log/TEMPLATE-week2-gate.md` diagnostic checklist |

---

## Credits

Project team: MSML635 Group 5 (Members 1, 2, 3 — assignment in `docs/team.md`).
Builds on GRACE (Zhao et al. 2026) + verl + Qwen3 + HotpotQA + QASPER.
