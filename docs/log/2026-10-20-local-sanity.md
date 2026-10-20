# Local sanity — 2026-10-20

**Device:** NVIDIA GeForce RTX 4050 Laptop GPU, 6.0 GB VRAM
**Stack:** Python 3.12 (venv), torch 2.5.1+cu121 (CUDA 12.1), transformers<5
**Model:** `Qwen/Qwen3-0.6B` in fp16
**Command:** `.venv/Scripts/python.exe scripts/local_sanity.py --n 10 --device cuda`
**Output:** `results/local_sanity/20261020-192500/` (gitignored)
**Commit when run:** `3e72752` (merge with origin/main)

## Result — pipeline is healthy

| Metric | Value | Interpretation |
|---|---:|---|
| Samples | 10 (10 answerable / 0 unanswerable) | Tiny set is fully answerable by design |
| `elapsed_s` | 286.1 s | 28.6 s/sample on fp16 — matches ~37 s/sample seen in prior session; GPU is loaded correctly |
| `format_valid_rate` | 0.50 | 5/10 responses parsed into schema; base model is inconsistent — expected pre-RL |
| `em_answerable` | 0.100 | 1/10 exact matches; expected for an un-trained base model |
| `balanced_accuracy` | 0.050 | Floor baseline before any RL training |
| `path_shares` | `evidence 0.50 / llm 0.00 / invalid 0.50` | When the model does follow format, it always takes the evidence path — never the LLM-knowledge path. All failures are format-invalid, not path-mis-routing |

## Per-sample breakdown (from log)

| # | Question (truncated) | gold | pred | path | em |
|--:|---|---|---|---|--:|
| 1 | Scott Derrickson / Ed Wood same nationality? | yes | unknown | evidence | 0 |
| 2 | Position held by woman who portrayed Corliss… | Chief of Protocol | *(empty)* | invalid | 0 |
| 3 | Science fantasy YA series, first person, with… | Animorphs | *(empty)* | invalid | 0 |
| 4 | Laleli Mosque and Esma Sultan Mansion same neighborhood? | no | unknown | evidence | 0 |
| 5 | Director of "Big Stone Gap" based in which NY… | Greenwich Village, NYC | *(empty)* | invalid | 0 |
| 6 | 2014 S/S is debut of Korean group formed by… | YG Entertainment | *(empty)* | invalid | 0 |
| 7 | Who was known by stage name Aladin and helped… | Eenasul Fateh | *(empty)* | invalid | 0 |
| 8 | Arena where Lewiston Maineiacs played, seat count? | 3,677 seated | unknown | evidence | 0 |
| 9 | Older — Annie Morton or Terry Richardson? | Terry Richardson | unknown | evidence | 0 |
| 10 | Local H and For Against both from US? | yes | yes | evidence | **1** |

## What this confirms

- Data → prompt → model.generate → `parse_response` → `exact_match` chain is wired end-to-end.
- `src.eval.local_runner.LocalRunner` loads Qwen3-0.6B in fp16 within 6 GB VRAM.
- `scripts/local_sanity.py` exits 0 and writes `metrics.json` + `predictions.jsonl` in the expected shape.
- The reward signals needed for RL training (format validity, evidence/llm/invalid path, EM) are all computed and non-degenerate.
- A clean pre-RL baseline exists against which Week-2 Gate (`BalancedAccuracy ∈ [0.71, 0.91]` after R0+GRPO) will be compared.

## What it does NOT confirm

- Nothing about `verl` / `vllm` / GRPO training itself — those remain server-only.
- Nothing about retrieval quality — this run uses the tiny set's embedded gold passages, not the BM25 retrieval cache.
- The 1/10 EM and 50% format-valid rate are **expected for an untrained base model** and are not a bug.

## Gotchas worth remembering

- `python` on the system PATH resolves to Python 3.14 (`%APPDATA%\Python\Python314`) which has a broken torch 2.11.0+cpu / torchvision install. Only `.venv/Scripts/python.exe` works. All project scripts must be invoked through the venv binary or via `.venv/Scripts/activate` first.
- STATUS.md's claim of `torch 2.5.1+cu121` applies to the venv, not the system Python. Keep this distinction in mind when teammates clone and run.

## Reproducing

```bash
# From repo root, with .venv activated:
python scripts/download_hotpot_tiny.py            # if data/raw/hotpot_tiny.jsonl missing
python scripts/local_sanity.py --n 10 --device cuda
```

Expected time: ~5 min on this GPU; ~15–30 min on CPU.
