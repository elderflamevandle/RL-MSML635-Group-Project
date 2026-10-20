# Local sanity — 2026-10-20

**Device:** NVIDIA GeForce RTX 4050 Laptop GPU, 6.0 GB VRAM
**Stack:** Python 3.12 (`.venv`), torch 2.5.1+cu121 (CUDA 12.1), transformers<5
**Model:** `Qwen/Qwen3-0.6B` in fp16
**Commit when run:** `f4cc09a` (after merge with origin/main)

Two runs on this date against `data/raw/hotpot_tiny.jsonl` (50 HotpotQA samples, all answerable by construction). First was a 10-sample smoke test; the 50-sample run is the **authoritative pre-RL baseline** reported below.

## Authoritative baseline — 50 samples

```
.venv/Scripts/python.exe scripts/local_sanity.py --n 50 --device cuda
```

**Output:** `results/local_sanity/20261020-194948/` (gitignored)

| Metric | Value |
|---|---:|
| `n` / `n_answerable` | 50 / 50 |
| `elapsed_s` | 1273.3 s (**25.5 s/sample**) |
| `format_valid_rate` | **0.70** |
| `em_answerable` | **0.080** (4/50 exact matches) |
| `abstention_unanswerable` | 0.00 (undefined — tiny set is all-answerable) |
| `balanced_accuracy` | **0.040** |
| `path_shares.evidence` | 0.68 |
| `path_shares.llm` | 0.02 |
| `path_shares.invalid` | 0.30 |

**Floor to beat.** This is the Qwen3-0.6B un-trained baseline against which the Week-2 Gate (`BalancedAccuracy ∈ [0.71, 0.91]` after R0+GRPO on server) will be judged.

## 10-sample smoke test (earlier same day)

**Output:** `results/local_sanity/20261020-192500/`

| Metric | Value |
|---|---:|
| `n` | 10 / 10 answerable |
| `elapsed_s` | 286.1 s (28.6 s/sample) |
| `format_valid_rate` | 0.50 |
| `em_answerable` | 0.100 (1/10) |
| `balanced_accuracy` | 0.050 |
| `path_shares` | evidence 0.50 / llm 0.00 / invalid 0.50 |

The 50-sample numbers are more trustworthy; the 10-sample run was a smoke test to confirm GPU + model loading before committing to the longer run.

## What this confirms

- Data → prompt → `model.generate` → `parse_response` → `exact_match` chain is wired end-to-end on this stack.
- `src.eval.local_runner.LocalRunner` loads Qwen3-0.6B in fp16 within 6 GB VRAM.
- `scripts/local_sanity.py` exits 0 and writes `metrics.json` + `predictions.jsonl` in the expected shape.
- All reward signals used by R0/R1/R2 downstream (format validity, path routing, EM) compute without crashing.
- Base model's natural failure mode on this prompt schema is **invalid-format** (30% of outputs) rather than wrong-path routing — RL training's first job is to drive the invalid share down.

## What it does NOT confirm

- Nothing about `verl` / `vllm` / GRPO training itself — those remain server-only.
- Nothing about retrieval quality — this run uses the tiny set's embedded gold passages, not the BM25 retrieval cache.
- `abstention_unanswerable` is undefined here because the tiny set has no unanswerable samples — real abstention metrics require a balanced split (the `data/cache/bm25_*.jsonl` files produced earlier the same day).
- Low EM at this stage is **expected for an untrained base model** and is not a bug.

## Gotchas worth remembering

- `python` on the Windows PATH resolves to Python 3.14 (`%APPDATA%\Python\Python314`) which has a broken torch 2.11.0+cpu / torchvision install. Only `.venv/Scripts/python.exe` works. All project scripts must be invoked through the venv binary (or via `.venv/Scripts/activate` first).
- STATUS.md's claim of `torch 2.5.1+cu121` applies to the venv only. Keep this distinction in mind when teammates clone and run.
- One `top_k` generation flag warning + one `torch_dtype` deprecation warning fire from the HF stack — harmless under the pinned `transformers<5`.

## Reproducing

```bash
# From repo root, with .venv activated:
python scripts/download_hotpot_tiny.py            # if data/raw/hotpot_tiny.jsonl missing
python scripts/local_sanity.py --n 50 --device cuda
```

Expected wall time: ~21 min on this GPU. Allow ~15–30 min on CPU-only machines.
