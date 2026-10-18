# Project Handoff — Scaffolding Complete

**Generated:** 2026-10-11 (Day 1 of 7-week project)
**Scaffold scope:** Plan Tasks 0.2, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3 (Weeks 0-2 laptop-safe work)
**Test state:** `37 passed, 1 deselected` (the deselected one is the slow dense-retriever test — runs on server with the embedding model)

---

## What's done

### Repo skeleton (Plan Task 0.2)
- `src/` with 7 subpackages (`common`, `data`, `prompts`, `rewards`, `eval`, `training`, `experiments`) — all import cleanly.
- `pyproject.toml` with runtime deps + `[dev]` and `[train]` extras. `[train]` (vllm, accelerate, peft, wandb) is **server-only** — don't install on laptops.
- `.gitignore`, `README.md`, `configs/`, `scripts/`, `tests/`, `docs/{log,figures,report}`, `data/{raw,cache}`, `results/` all in place.

### Shared integration contract (Plan Task 1.2)
- `src/common/types.py` — `Sample` dataclass (slots=True), `ParsedResponse` dataclass, `ResponsePath = Literal["evidence","llm","invalid"]` (renamed from `Path` to avoid shadowing `pathlib.Path`).
- `src/common/parse.py` — `parse_response(text) -> ParsedResponse` single-source-of-truth parser for model outputs.
- `src/data/hotpot.py` — `load_hotpot_tiny(path: str | Path)` loader with fail-loud assertion on empty gold passages.
- `data/raw/hotpot_tiny.jsonl` — 50 real HotpotQA samples (via `scripts/download_hotpot_tiny.py`).
- Tests: `test_types.py (1)`, `test_parse.py (5)`, `test_hotpot.py (1)` — all pass.

### Prompts + R0 reward + ROUGE-L metrics (Plan Task 1.3)
- `src/prompts/templates.py` — `RAG_SYSTEM` constant, `build_prompt(question, passages)` — single source of truth for prompt format.
- `src/rewards/metrics.py` — `rouge_l(pred, ref|list[ref])` with multi-ref max + zero guards; `quote_in_passages(quote, passages)` (used by R1/R2, documented).
- `src/rewards/r0_grace.py` — `R0Config` (frozen dataclass) + `r0_grace(response, sample, cfg) -> (float, dict)`. Branch logic matches GRACE exactly. `parts` dict pre-populated for logging; `correct_path = None` on invalid format (distinguishable from `False` on valid-but-wrong-path).
- Tests: `test_metrics.py (7)`, `test_r0.py (9)` — covers all 4 formula branches + hallucination documentation + edge cases.

### Training launcher + run naming + results convention (Plan Task 1.4)
- `src/training/run_naming.py` — `make_run_id(reward, optimizer, model, seed)` producing `<reward>_<optimizer>_<model>_s<seed>_<YYYYMMDD-HHMMSS>`.
- `scripts/launch_run.sh` — **server-only** launcher that creates `results/<run_id>/` and tees training logs. Expects `~/rl-rag-venv` activated.
- `scripts/setup_env.sh` — **server-only** env setup. Has placeholder `VERL_COMMIT="FILL_FROM_GRACE_README_ON_DAY_1"` — fill in once you locate the GRACE paper's reference verl commit.
- `configs/skeleton.yaml` — walking-skeleton 1-step GRPO config for the Week 1 Task 1.1 end-to-end bringup.
- `docs/results-convention.md` — layout docs for `results/<run_id>/`.
- Tests: `test_run_naming.py (2)` — passes.

### Retrievers + balanced split + cache (Plan Task 2.1)
- `src/data/retrievers.py` — `BM25Retriever` (uses `rank_bm25` + regex `\w+` tokenizer so punctuation is stripped) and `DenseRetriever` (uses `sentence_transformers`, Qwen3-Embedding-0.6B). Both: stable argsort + NaN guard for reproducibility.
- `src/data/balanced_split.py` — `build_balanced_split(samples, retrieve_fn, k=5)` preserving all Sample fields except `retrieved_passages` and `is_answerable`.
- `src/data/cache.py` — `RetrievalCache(dir)` with SHA-256 keys, **atomic writes** (temp + rename), and **collision detection** (stores full key tuple, raises on mismatch).
- `scripts/download_hotpot_full.py` — **server-only** downloader for 2000 train + 500 val HotpotQA samples.
- `scripts/build_retrieval.py` — **server-only** orchestrator; now actually wires in `RetrievalCache` (was dead code pre-review), and emits a `prompt` field for verl's dataloader. Includes sanity assertion that `0.1 < ans_rate < 0.95` to catch silent all-unanswerable splits.
- Tests: `test_retrievers.py (3, 1 slow)`, `test_balanced_split.py (2)`, `test_cache.py (2)` — pass.

### R0 edge-case tests + reward design doc (Plan Task 2.2)
- `tests/test_r0.py` — now 9 tests. New edge cases: question-copy, zero-overlap evidence, llm-path-on-answerable.
- `docs/rewards.md` — implementation-level reference for R0, R1 (to be implemented W3), R2 (to be implemented W3), audit utilities, hyperparameter starting values with "no sweeps until W6" tuning policy.

### Eval harness skeleton (Plan Task 2.3)
- `src/eval/metrics.py` — `exact_match(pred, ref)` (case + punctuation + whitespace normalized), `abstention_accuracy(predictions)` (TNR on unanswerable), `balanced_accuracy(em_ans, abst_unans)`.
- `src/eval/evaluate.py` — CLI entrypoint, **import-safe on laptop** (vllm/transformers imports are lazy inside `_generate`). Produces `metrics.json` + `predictions.jsonl` (includes raw_response for Week-4 audit).
- `configs/r0_grpo.yaml` — full-size GRPO config pointing at `data/cache/dense_train.jsonl` and `src.rewards.r0_grace:r0_grace`. verl field names will need iteration against the installed verl version (Week 2 Task 2.3 handles this).
- Tests: `test_eval_metrics.py (6)` — pass.

---

## What's deferred to the server session

These require a GPU and/or heavy packages (`vllm`, `verl`, `torch` with CUDA, `sentence-transformers` for the 1.2GB embedding model):

1. **Plan Task 0.1 — Team logistics.** Still needs:
   - Agree on who is Member 1 / 2 / 3.
   - Share Groq API key via password manager (verify with the curl snippet in Plan Task 0.1).
   - Decide results-sync mechanism (recommend NFS mount or rsync, not git-lfs).
   - Weekly sync meeting time.
   - Document all of this in `docs/team.md` (not yet created).

2. **Plan Task 1.1 — Walking skeleton end-to-end.**
   - On GPU server: run `scripts/setup_env.sh` after filling `VERL_COMMIT` from the GRACE paper's reference repo.
   - Then `bash scripts/launch_run.sh configs/skeleton.yaml` and iterate on YAML field names until verl accepts them.
   - Expected first-run failures (and fixes): config schema mismatch (read verl error, fix YAML), OOM (lower `gpu_memory_utilization` and `train_batch_size`), model-load failure (check HF token/cache).
   - Record working config + any adjustments in `docs/log/<date>-skeleton-run.md`.

3. **Plan Task 2.1 — Full retrieval build.**
   - `python scripts/download_hotpot_full.py` → downloads 2000+500 raw HotpotQA.
   - `python scripts/build_retrieval.py` → runs both retrievers, writes 4 files to `data/cache/`. Dense retriever downloads Qwen3-Embedding-0.6B first time (~1.2GB).
   - Verify observed ans:unans ratios in printed output; document in `docs/data.md`.
   - **Hard failure mode to watch:** the sanity assertion `0.1 < ans_rate < 0.95` will fire if gold-passage string matching silently breaks. If it fires, debug the string normalization in `_hotpot_row_to_sample` vs the retriever's indexed corpus.

4. **Plan Task 2.3 — First full R0+GRPO run (Week 2 gate).**
   - `bash scripts/launch_run.sh configs/r0_grpo.yaml` (~2-6h on single GPU).
   - Then `python -m src.eval.evaluate --checkpoint <path> --split data/cache/dense_val.jsonl --out results/<run_id>/metrics.json`.
   - **Week 2 Gate:** Balanced Accuracy must be in [0.71, 0.91] (within 10 points of GRACE's HotpotQA 0.81). If not — whole team halts and pair-debugs before Week 3.

---

## What the team should do first on Day 1

1. **Everyone:** clone this repo onto their laptop, run `python -m venv .venv && .venv/Scripts/activate && pip install -e ".[dev]"`, then `pytest -q`. Confirm **37 passed, 1 deselected** matches locally. If anything red, debug before anything else.
2. **Member 3 (training lead):** SSH to the GPU server, start Plan Task 0.1 (env setup — find the GRACE verl commit, fill `scripts/setup_env.sh`, run it).
3. **Member 1 (data lead):** read `src/common/types.py`, `src/data/hotpot.py`, `src/data/retrievers.py`, `src/data/cache.py` — you'll own the full retrieval build in Week 2.
4. **Member 2 (rewards lead):** read `src/rewards/r0_grace.py`, `src/rewards/metrics.py`, `docs/rewards.md` — you'll implement R1 and R2 in Week 3.
5. **All three together:** on the server, do Plan Task 1.1 Days 1-3 (walking skeleton pair-debug). This is the only task designed for all three in a room.

---

## Known caveats / open items

- **`VERL_COMMIT` placeholder in `scripts/setup_env.sh`** — must be filled from the GRACE paper on Day 1.
- **verl config field names in `configs/*.yaml`** — these are the field names mentioned in the GRACE paper's reference verl config. The installed verl version will probably want slightly different names; iterate against actual verl errors.
- **No git initialized** — user deferred. When ready, see Plan Task 0.0 (Optional). All scaffolding changes to date can be committed in one shot: `git init && git add . && git commit -m "feat: scaffold Weeks 0-2 (Tasks 0.2, 1.2-1.4, 2.1-2.3)"`.
- **`data/raw/hotpot_tiny.jsonl` is checked in?** — No, it's gitignored (per `.gitignore`'s `data/raw/`). Members will regenerate it with `python scripts/download_hotpot_tiny.py` after cloning.
- **Minor review follow-ups not blocking:** `_hotpot_row_to_sample` has no direct unit test (currently tested through `load_hotpot_tiny`); `test_parse_llm_path` + a round-trip test between `build_prompt` output and `parse_response` would tighten the prompt-contract. Fine to leave for a quiet moment.

---

## File inventory

```
src/
  __init__.py
  common/
    __init__.py
    parse.py              — parse_response() single source of truth
    types.py              — Sample, ParsedResponse, ResponsePath
  data/
    __init__.py
    balanced_split.py     — build_balanced_split()
    cache.py              — RetrievalCache (SHA-256 keys, atomic, collision-safe)
    hotpot.py             — load_hotpot_tiny() with fail-loud assertion
    retrievers.py         — BM25Retriever + DenseRetriever (stable argsort, NaN guard)
  eval/
    __init__.py
    evaluate.py           — CLI; lazy vllm/transformers imports
    metrics.py            — EM, abstention_accuracy, balanced_accuracy
  experiments/__init__.py  (empty — Week 4+)
  prompts/
    __init__.py
    templates.py          — RAG_SYSTEM + build_prompt()
  rewards/
    __init__.py
    metrics.py            — rouge_l + quote_in_passages
    r0_grace.py           — R0Config + r0_grace()
  training/
    __init__.py
    run_naming.py         — make_run_id()
configs/
  r0_grpo.yaml            — server-only; full-size GRPO
  skeleton.yaml           — server-only; 1-step walking skeleton
scripts/
  build_retrieval.py      — server-only
  download_hotpot_full.py — server-only
  download_hotpot_tiny.py — laptop-safe (50 samples)
  launch_run.sh           — server-only
  setup_env.sh            — server-only (fill VERL_COMMIT!)
tests/
  __init__.py
  test_balanced_split.py  (2)
  test_cache.py           (2)
  test_eval_metrics.py    (6)
  test_hotpot.py          (1)
  test_metrics.py         (7)
  test_parse.py           (5)
  test_r0.py              (9)
  test_retrievers.py      (3; 1 slow — deselected by default)
  test_run_naming.py      (2)
  test_types.py           (1)
docs/
  results-convention.md
  rewards.md
  log/                    (empty — fill as Day-1 debugging progresses)
  figures/                (empty — Week 5 fills this)
  report/                 (empty — Week 5 fills this)
  superpowers/
    plans/2026-10-11-rl-rag-reward-design.md
    specs/2026-10-11-rl-rag-reward-design-plan.md
pyproject.toml
.gitignore
README.md
HANDOFF.md               (this file)
```

---

**Final test state:**
```
$ pytest -q -m "not slow"
.....................................                                    [100%]
37 passed, 1 deselected in ~3.3s
```

Ready for the human team to pick up at Plan Task 0.1 (team logistics) and Plan Task 1.1 (walking skeleton on the server).
