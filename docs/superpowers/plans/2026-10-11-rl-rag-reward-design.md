# RL-Assisted RAG — Reward Design Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **For the human team:** work the tasks in order within each week. Each member picks up their lane after the shared Week-1 skeleton. Gates at the end of each week block the next week until satisfied.

**Spec:** `docs/superpowers/specs/2026-10-11-rl-rag-reward-design-plan.md`

**Goal:** Implement and compare three reward functions (R0 GRACE baseline, R1 cost-sensitive, R2 hard-gated) with two optimizers (GRPO, DAPO) on HotpotQA using verl + Qwen3-0.6B + LoRA, producing defensible answers to RQ1 (does gating stop hallucination?) and RQ2 (GRPO vs DAPO sample efficiency).

**Architecture:** A shared `src/common/` contract (`Sample` dataclass + `parse_response` function) connects three independently-developed lanes: data/eval (Member 1), reward design (Member 2), and training/experiments (Member 3). verl + vLLM provides the RL algorithms and rollouts verbatim; we add only reward functions, prompts, data processing, and evaluation on top. Dev happens on laptops with CPU-mocked tests; training happens on a dedicated GPU server.

**Tech Stack:** Python 3.12, CUDA 12.4, verl (pinned to GRACE commit), vLLM, PyTorch, HuggingFace `datasets` + `transformers` + `sentence-transformers`, `rank-bm25`, Qwen3-0.6B (main) / Qwen3-1.7B (hero), Groq Llama 3.3 70B API (judge), `pytest`, `rouge-score`, `bert-score`.

**Git note:** User deferred git initialization. Each task ends with a verification step (tests pass or run completes). When git is initialized later (see Task 0.0 Optional), the team can retroactively commit using the suggested commit messages included at the end of each task.

---

## Table of contents

- [Week 0 (pre-flight): shared setup](#week-0-preflight--shared-setup)
- [Week 1 (Oct 11–18): walking skeleton + env setup](#week-1-oct-1118--walking-skeleton--env-setup)
- [Week 2 (Oct 19–25): full HotpotQA + R0 reproduction](#week-2-oct-1925--full-hotpotqa--r0-reproduction)
- [Week 3 (Oct 26–Nov 1): reward variants + Groq LJ + DAPO](#week-3-oct-26nov-1--reward-variants--groq-lj--dapo)
- [Week 4 (Nov 2–8): full grid execution + first analysis](#week-4-nov-28--full-grid-execution--first-analysis)
- [Week 5 (Nov 9–15): should-runs + hero run + report draft](#week-5-nov-915--shouldruns--hero-run--report-draft)
- [Week 6 (Nov 16–22): buffer / stretch / ablations](#week-6-nov-1622--buffer--stretch--ablations)
- [Week 7 (Nov 23–29): final polish + submit](#week-7-nov-2329--final-polish--submit)

---

## Week 0 (pre-flight) — Shared setup

Goals before Week 1 kickoff: everyone has their dev env, the repo has structure, and the team agrees on sync mechanism.

### Task 0.1: Agree on team logistics (whole team, 30 min)

- [ ] **Step 1:** Decide and record in `docs/team.md`:
  - Who is Member 1 / Member 2 / Member 3 (initial assignment, swappable)
  - GPU server access details (hostname, how to SSH, who has access)
  - Groq API key — one member creates account, shares key via password manager
  - W&B team account OR local-tensorboard-only choice
  - Results sync mechanism — pick one of: shared NFS mount, rsync + cron, or git-lfs (recommend NFS or rsync; git-lfs adds friction)
  - Weekly sync meeting time
- [ ] **Step 2:** Verify everyone can SSH to the GPU server
- [ ] **Step 3:** Verify Groq API key works with a `curl` test:

```bash
curl https://api.groq.com/openai/v1/chat/completions \
  -H "Authorization: Bearer $GROQ_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":"ping"}]}'
```

Expected: JSON response with a short reply. If 401, re-check key.

### Task 0.2: Initialize repo structure (one member, 20 min)

**Files:**
- Create: `F:/Reinforcement_Learning_Project/.gitignore`
- Create: `F:/Reinforcement_Learning_Project/README.md`
- Create: `F:/Reinforcement_Learning_Project/pyproject.toml`
- Create: `F:/Reinforcement_Learning_Project/src/__init__.py` and subpackage `__init__.py` files

- [ ] **Step 1:** Create the directory skeleton:

```bash
cd F:/Reinforcement_Learning_Project
mkdir -p src/{data,prompts,rewards,eval,training,experiments,common}
mkdir -p configs scripts tests docs/log results data/raw data/cache
touch src/__init__.py
for d in data prompts rewards eval training experiments common; do touch src/$d/__init__.py; done
touch tests/__init__.py
```

- [ ] **Step 2:** Write `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/
.venv/
venv/

# Project
results/
data/cache/
data/raw/
*.ckpt
*.pt
*.safetensors
wandb/
.env

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 3:** Write `pyproject.toml`:

```toml
[project]
name = "rl-rag"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "datasets>=2.20",
    "transformers>=4.44",
    "sentence-transformers>=3.0",
    "rank-bm25>=0.2.2",
    "rouge-score>=0.1.2",
    "bert-score>=0.3.13",
    "torch>=2.3",
    "numpy",
    "pydantic>=2",
    "pyyaml",
    "requests",
    "tenacity",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov", "ruff", "mypy"]
train = ["vllm", "accelerate", "peft", "wandb"]  # heavy; install on server only

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 4:** Write minimal `README.md`:

```markdown
# RL-Assisted RAG — Reward Design

MSML635 Group 5. Spec: `docs/superpowers/specs/2026-10-11-rl-rag-reward-design-plan.md`. Plan: `docs/superpowers/plans/2026-10-11-rl-rag-reward-design.md`.

## Setup (laptop dev)
```
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -e ".[dev]"
pytest -q
```

## Setup (GPU server)
See `scripts/setup_env.sh`.
```

- [ ] **Step 5:** Verify structure:

```bash
find src tests docs configs scripts -type d | sort
```

Expected output includes all 7 `src/*/` directories plus `tests`, `docs`, `configs`, `scripts`.

### Task 0.0 (optional): Initialize git

Only when the team agrees. Deferred by user for now.

- [ ] **Step 1:** `git init`
- [ ] **Step 2:** `git add .gitignore README.md pyproject.toml src/ tests/ docs/ configs/ scripts/`
- [ ] **Step 3:** `git commit -m "chore: initial repo skeleton"`
- [ ] **Step 4:** Add a remote (GitHub classroom repo, personal, etc.) and push.

---

## Week 1 (Oct 11–18) — Walking skeleton + env setup

**Week goal:** one tiny end-to-end run completes on server; all members have working dev loops.

### Task 1.1 (together, Days 1-3): Walking-skeleton end-to-end run

Everyone pairs on this. No fan-out until it works.

**Files:**
- Create: `scripts/setup_env.sh`
- Create: `scripts/download_hotpot_tiny.py`
- Create: `src/prompts/templates.py` (minimal)
- Create: `src/rewards/r0_grace.py` (hardcoded minimal version)
- Create: `configs/skeleton.yaml`
- Create: `scripts/launch_skeleton.sh`

- [ ] **Step 1:** Create `scripts/setup_env.sh` to install verl on the server:

```bash
#!/usr/bin/env bash
# Server-side env setup. Run on the GPU box.
set -euo pipefail

PY=python3.12
VERL_COMMIT="<FILL FROM GRACE README>"   # pin to GRACE's verl commit

$PY -m venv ~/rl-rag-venv
source ~/rl-rag-venv/bin/activate
pip install --upgrade pip

# Project deps
pip install -e ".[dev,train]"

# verl at pinned commit
pip install "git+https://github.com/volcengine/verl.git@${VERL_COMMIT}"

# Verify CUDA
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"

# Verify vllm
python -c "import vllm; print('vllm', vllm.__version__)"

echo "Setup OK."
```

Replace `<FILL FROM GRACE README>` with the exact commit hash from the GRACE paper's reference repo (one of the first things to find on Day 1).

- [ ] **Step 2:** Run `setup_env.sh` on the GPU server. Expected: ends with "Setup OK." and `torch.cuda.is_available()` prints `True`. If vllm install fails, read the error — typically CUDA version mismatch; fix by picking the vllm version matching CUDA 12.4.

- [ ] **Step 3:** Create `scripts/download_hotpot_tiny.py` to grab 50 samples:

```python
"""Download first 50 HotpotQA samples for the walking skeleton."""
from datasets import load_dataset
from pathlib import Path
import json

OUT = Path("data/raw/hotpot_tiny.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset("hotpot_qa", "distractor", split="validation[:50]")
with OUT.open("w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps(row) + "\n")

print(f"Wrote {OUT} with 50 samples.")
```

Run it: `python scripts/download_hotpot_tiny.py`. Expected: file exists with 50 lines.

- [ ] **Step 4:** Create `src/prompts/templates.py` (minimal):

```python
"""Prompt templates. SINGLE SOURCE OF TRUTH for prompt format — both rewards
and training import from here. Do not inline prompt strings elsewhere."""

RAG_SYSTEM = (
    "You are a careful RAG assistant. Read the passages. "
    "If the passages contain the answer, respond with "
    "<evidence>exact quoted sentence(s)</evidence>"
    "<answer>short answer</answer>. "
    "If the passages do NOT contain the answer, respond with "
    "<llm>I cannot answer from the provided evidence.</llm>"
    "<answer>unknown</answer>."
)

def build_prompt(question: str, passages: list[str]) -> str:
    numbered = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    return (
        f"{RAG_SYSTEM}\n\n"
        f"Passages:\n{numbered}\n\n"
        f"Question: {question}\n\n"
        f"Response:"
    )
```

- [ ] **Step 5:** Create `src/rewards/r0_grace.py` (minimal hardcoded version — will be rewritten properly in Week 2):

```python
"""Hardcoded minimal R0 for the walking skeleton. Will be replaced in Week 2."""
import re

def r0_grace_skeleton(response: str, gold_answer: str, is_answerable: bool) -> float:
    has_evidence = "<evidence>" in response and "</evidence>" in response
    has_llm = "<llm>" in response and "</llm>" in response
    has_answer = "<answer>" in response and "</answer>" in response
    valid_format = has_answer and (has_evidence ^ has_llm)
    if not valid_format:
        return 0.0

    chose_evidence = has_evidence
    correct_path = (chose_evidence and is_answerable) or ((not chose_evidence) and not is_answerable)
    rf = 0.5
    rp = 1.0 if correct_path else 0.0

    if not correct_path:
        return rf

    if chose_evidence:
        m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        answer = m.group(1).strip() if m else ""
        rouge_like = 1.0 if gold_answer.lower() in answer.lower() else 0.0
        return rf + rp + 0.5 * rouge_like  # α=β=0.5·x placeholder
    return rf + rp
```

- [ ] **Step 6:** Create `configs/skeleton.yaml` using verl's GRPO config shape (consult the GRACE repo for the exact field names; the shape below is illustrative):

```yaml
# Walking-skeleton GRPO config. Smallest workable configuration.
trainer:
  project_name: rl-rag-skeleton
  experiment_name: skeleton
  n_gpus_per_node: 1
  nnodes: 1
  total_training_steps: 1
  save_freq: 1

data:
  train_files: data/raw/hotpot_tiny.jsonl
  val_files: data/raw/hotpot_tiny.jsonl
  prompt_key: prompt
  max_prompt_length: 2048
  max_response_length: 512
  train_batch_size: 4

actor_rollout_ref:
  model:
    path: Qwen/Qwen3-0.6B
    enable_lora: true
    lora_rank: 16
  rollout:
    name: vllm
    gpu_memory_utilization: 0.6
    n: 4   # G=4 rollouts for GRPO group

algorithm:
  name: grpo
  kl_coef: 0.001

reward_fn:
  path: src.rewards.r0_grace:r0_grace_skeleton
```

(Field names will need adjustment to match verl's actual config schema — look at `verl/examples/` for working templates when installing.)

- [ ] **Step 7:** Create `scripts/launch_skeleton.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source ~/rl-rag-venv/bin/activate
cd $(dirname $0)/..
python -m verl.trainer.main_ppo --config configs/skeleton.yaml
```

(The exact verl entrypoint command comes from the GRACE repo's launch scripts.)

- [ ] **Step 8:** Run `bash scripts/launch_skeleton.sh` on the server. Expected: a single training step completes, you see rollout outputs printed, loss printed, and no exception. If it fails, pair-debug — the fail-modes bucket into (a) config schema mismatch (read verl error, fix YAML), (b) OOM (lower `gpu_memory_utilization` and `train_batch_size`), (c) model load failure (check HF cache, token access).

- [ ] **Step 9:** Record the working command and any config adjustments in `docs/log/2026-10-13-skeleton-run.md`:

```markdown
# Walking-skeleton run — <date>

Config used: configs/skeleton.yaml (version at this commit)
Command: bash scripts/launch_skeleton.sh
Output location: results/<run_id>/

Verl commit: <hash>
GPU: <name>
Peak memory: <GB>
Wall time: <seconds>

Issues hit and fixes:
- ...
```

- [ ] **Step 10 (checkpoint):** All three members confirm they saw the training step complete. If not — do not fan out yet.

**Suggested commit message when git is initialized:** `feat(skeleton): working end-to-end 1-step GRPO run on 50 HotpotQA samples`

---

### Task 1.2 (Member 1, Days 4-7): Scaffold `src/common/` and `src/data/`

**Files:**
- Create: `src/common/types.py`
- Create: `src/common/parse.py`
- Create: `src/data/hotpot.py`
- Create: `tests/test_types.py`
- Create: `tests/test_parse.py`
- Create: `tests/test_hotpot.py`

- [ ] **Step 1 (TDD): Write failing test for `Sample` dataclass** — `tests/test_types.py`:

```python
from src.common.types import Sample

def test_sample_minimal_fields():
    s = Sample(
        sample_id="hp_1",
        question="Who directed Inception?",
        gold_passages=["Christopher Nolan directed Inception in 2010."],
        distractor_passages=["Stanley Kubrick directed many films."],
        gold_answer="Christopher Nolan",
        gold_evidence=["Christopher Nolan directed Inception in 2010."],
        retrieved_passages=["Christopher Nolan directed Inception in 2010."],
        is_answerable=True,
    )
    assert s.sample_id == "hp_1"
    assert s.is_answerable is True
    assert len(s.retrieved_passages) == 1
```

- [ ] **Step 2:** Run `pytest tests/test_types.py -v`. Expected: `ImportError: cannot import name 'Sample'`.

- [ ] **Step 3:** Implement `src/common/types.py`:

```python
"""Shared types used by every lane. Imported from rewards, eval, training."""
from dataclasses import dataclass, field
from typing import Literal

@dataclass(slots=True)
class Sample:
    sample_id: str
    question: str
    gold_passages: list[str]
    distractor_passages: list[str]
    gold_answer: str
    gold_evidence: list[str]
    retrieved_passages: list[str]
    is_answerable: bool

Path = Literal["evidence", "llm", "invalid"]

@dataclass(slots=True)
class ParsedResponse:
    path: Path
    quoted_evidence: list[str] = field(default_factory=list)
    answer: str = ""
    format_valid: bool = False
    raw: str = ""
```

- [ ] **Step 4:** `pytest tests/test_types.py -v`. Expected: PASS.

- [ ] **Step 5 (TDD): Write failing tests for `parse_response`** — `tests/test_parse.py`:

```python
from src.common.parse import parse_response

def test_parse_evidence_path():
    r = parse_response(
        "<evidence>Nolan directed Inception.</evidence><answer>Christopher Nolan</answer>"
    )
    assert r.path == "evidence"
    assert r.format_valid is True
    assert r.quoted_evidence == ["Nolan directed Inception."]
    assert r.answer == "Christopher Nolan"

def test_parse_llm_path():
    r = parse_response("<llm>Not in evidence.</llm><answer>unknown</answer>")
    assert r.path == "llm"
    assert r.format_valid is True
    assert r.answer == "unknown"

def test_parse_invalid_both_paths():
    r = parse_response("<evidence>x</evidence><llm>y</llm><answer>z</answer>")
    assert r.path == "invalid"
    assert r.format_valid is False

def test_parse_invalid_missing_answer():
    r = parse_response("<evidence>x</evidence>")
    assert r.path == "invalid"
    assert r.format_valid is False

def test_parse_multiple_evidence_quotes():
    r = parse_response(
        "<evidence>Sentence one.</evidence>"
        "<evidence>Sentence two.</evidence>"
        "<answer>A</answer>"
    )
    assert r.path == "evidence"
    assert r.quoted_evidence == ["Sentence one.", "Sentence two."]
    assert r.format_valid is True
```

- [ ] **Step 6:** `pytest tests/test_parse.py -v`. Expected: `ImportError`.

- [ ] **Step 7:** Implement `src/common/parse.py`:

```python
"""Response parser. SINGLE SOURCE OF TRUTH for interpreting model outputs.
Both reward functions and the eval harness call this — fix bugs here, not downstream."""
import re
from src.common.types import ParsedResponse

_EVIDENCE_RE = re.compile(r"<evidence>(.*?)</evidence>", re.DOTALL)
_LLM_RE = re.compile(r"<llm>(.*?)</llm>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

def parse_response(text: str) -> ParsedResponse:
    evidence_matches = _EVIDENCE_RE.findall(text)
    llm_matches = _LLM_RE.findall(text)
    answer_match = _ANSWER_RE.search(text)

    has_evidence = len(evidence_matches) > 0
    has_llm = len(llm_matches) > 0
    has_answer = answer_match is not None

    # Exactly one path tag, and an answer tag, must be present.
    one_path = has_evidence ^ has_llm
    format_valid = one_path and has_answer

    if not format_valid:
        return ParsedResponse(path="invalid", raw=text)

    answer = answer_match.group(1).strip()
    if has_evidence:
        return ParsedResponse(
            path="evidence",
            quoted_evidence=[q.strip() for q in evidence_matches],
            answer=answer,
            format_valid=True,
            raw=text,
        )
    return ParsedResponse(
        path="llm",
        answer=answer,
        format_valid=True,
        raw=text,
    )
```

- [ ] **Step 8:** `pytest tests/test_parse.py -v`. Expected: all 5 PASS.

- [ ] **Step 9 (TDD): Write failing test for HotpotQA loader** — `tests/test_hotpot.py`:

```python
from pathlib import Path
from src.data.hotpot import load_hotpot_tiny

def test_load_hotpot_tiny_returns_samples(tmp_path, monkeypatch):
    # Point loader at the committed tiny file.
    samples = load_hotpot_tiny(Path("data/raw/hotpot_tiny.jsonl"))
    assert len(samples) == 50
    s = samples[0]
    assert s.sample_id
    assert s.question
    assert len(s.gold_passages) > 0
    assert s.gold_answer
    # Each raw HotpotQA sample has gold support facts; is_answerable
    # defaults to True until Member 1 adds the distractor-swap logic in W2.
    assert isinstance(s.is_answerable, bool)
```

- [ ] **Step 10:** `pytest tests/test_hotpot.py -v`. Expected: `ImportError`.

- [ ] **Step 11:** Implement `src/data/hotpot.py`:

```python
"""HotpotQA loader. Converts raw HF rows into `Sample` objects.

In Week 1 this just maps raw fields through. The balanced ans/unans split and
retrieval are added in Week 2 by `src/data/balanced_split.py` and
`src/data/retrievers.py`.
"""
from pathlib import Path
import json
from src.common.types import Sample

def _hotpot_row_to_sample(row: dict) -> Sample:
    # HotpotQA distractor format: context = {"title": [...], "sentences": [[...], ...]}
    titles = row["context"]["title"]
    sentences = row["context"]["sentences"]
    passages = [" ".join(s) for s in sentences]

    supporting_titles = set(row["supporting_facts"]["title"])
    gold_passages = [
        passages[i] for i, t in enumerate(titles) if t in supporting_titles
    ]
    distractor_passages = [
        passages[i] for i, t in enumerate(titles) if t not in supporting_titles
    ]
    # Gold evidence sentences
    gold_evidence = []
    for title, sent_idx in zip(
        row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]
    ):
        if title in titles:
            para_idx = titles.index(title)
            if sent_idx < len(sentences[para_idx]):
                gold_evidence.append(sentences[para_idx][sent_idx])

    return Sample(
        sample_id=row["id"],
        question=row["question"],
        gold_passages=gold_passages,
        distractor_passages=distractor_passages,
        gold_answer=row["answer"],
        gold_evidence=gold_evidence,
        retrieved_passages=gold_passages + distractor_passages,  # placeholder: Week 2 adds real retrieval
        is_answerable=True,  # placeholder: Week 2's balanced-split logic sets this properly
    )

def load_hotpot_tiny(path: Path) -> list[Sample]:
    with path.open("r", encoding="utf-8") as f:
        return [_hotpot_row_to_sample(json.loads(line)) for line in f]
```

- [ ] **Step 12:** `pytest tests/test_hotpot.py -v`. Expected: PASS.

- [ ] **Step 13 (checkpoint):** Member 1's lane green. Suggested commit: `feat(common,data): shared Sample/parse_response + HotpotQA raw loader`

---

### Task 1.3 (Member 2, Days 4-7): Formalize prompts + R0 reward with unit tests

**Files:**
- Modify: `src/prompts/templates.py` (promote from Week 1.1 skeleton)
- Create: `src/rewards/metrics.py`
- Modify: `src/rewards/r0_grace.py` (full version)
- Create: `tests/test_r0.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1 (TDD): Write failing test for ROUGE-L wrapper** — `tests/test_metrics.py`:

```python
from src.rewards.metrics import rouge_l

def test_rouge_l_exact_match():
    assert rouge_l("the cat sat", "the cat sat") == 1.0

def test_rouge_l_zero_overlap():
    assert rouge_l("the cat sat", "dogs run fast") < 0.2

def test_rouge_l_partial():
    score = rouge_l("the cat sat on the mat", "cat sat on mat")
    assert 0.5 < score < 1.0

def test_rouge_l_against_multi_reference_max():
    score = rouge_l("the cat", ["dogs run", "the cat sat"])
    assert score > 0.5  # takes the max over references
```

- [ ] **Step 2:** `pytest tests/test_metrics.py -v`. Expected: `ImportError`.

- [ ] **Step 3:** Implement `src/rewards/metrics.py`:

```python
"""Reward-component metrics. ROUGE-L here matches what GRACE uses."""
from rouge_score import rouge_scorer

_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def rouge_l(prediction: str, reference: str | list[str]) -> float:
    """ROUGE-L F1. If reference is a list, returns the max over references."""
    if isinstance(reference, str):
        references = [reference]
    else:
        references = reference
    if not prediction or not references:
        return 0.0
    scores = [
        _SCORER.score(ref, prediction)["rougeL"].fmeasure for ref in references
    ]
    return max(scores)

def quote_in_passages(quote: str, passages: list[str]) -> bool:
    """True if `quote` appears as a substring in any passage (case-sensitive)."""
    return any(quote.strip() in p for p in passages)
```

- [ ] **Step 4:** `pytest tests/test_metrics.py -v`. Expected: all 4 PASS.

- [ ] **Step 5 (TDD): Write failing tests for R0** — `tests/test_r0.py`:

```python
from src.common.types import Sample
from src.rewards.r0_grace import r0_grace, R0Config

def _sample_ans(**overrides) -> Sample:
    base = dict(
        sample_id="s", question="q",
        gold_passages=["Christopher Nolan directed Inception."],
        distractor_passages=["Kubrick made 2001."],
        gold_answer="Christopher Nolan",
        gold_evidence=["Christopher Nolan directed Inception."],
        retrieved_passages=["Christopher Nolan directed Inception.", "Kubrick made 2001."],
        is_answerable=True,
    )
    base.update(overrides)
    return Sample(**base)

def _sample_unans(**overrides) -> Sample:
    return _sample_ans(
        gold_passages=[],  # replaced
        retrieved_passages=["Kubrick made 2001.", "Spielberg made Jaws."],
        is_answerable=False,
        **overrides,
    )

CFG = R0Config(Rf=0.5, Rp=1.0, alpha=0.5, beta=0.5)

def test_r0_invalid_format_scores_zero():
    r, parts = r0_grace("no tags here", _sample_ans(), CFG)
    assert r == 0.0
    assert parts["format_valid"] is False

def test_r0_valid_format_wrong_path_scores_rf_only():
    # Answerable sample but model abstains → wrong path.
    resp = "<llm>cannot answer</llm><answer>unknown</answer>"
    r, parts = r0_grace(resp, _sample_ans(), CFG)
    assert r == 0.5
    assert parts["correct_path"] is False

def test_r0_correct_llm_path_scores_rf_plus_rp():
    # Unanswerable sample, model abstains → correct <llm>.
    resp = "<llm>cannot answer</llm><answer>unknown</answer>"
    r, parts = r0_grace(resp, _sample_unans(), CFG)
    assert r == 1.5

def test_r0_correct_evidence_path_full_credit():
    resp = (
        "<evidence>Christopher Nolan directed Inception.</evidence>"
        "<answer>Christopher Nolan</answer>"
    )
    r, parts = r0_grace(resp, _sample_ans(), CFG)
    assert r == 0.5 + 1.0 + 0.5 * 1.0 + 0.5 * 1.0

def test_r0_correct_evidence_path_wrong_quote_partial_credit():
    resp = (
        "<evidence>Something irrelevant about Jaws.</evidence>"
        "<answer>Christopher Nolan</answer>"
    )
    r, parts = r0_grace(resp, _sample_ans(), CFG)
    # Rf + Rp + α·0 (low RLev) + β·1.0 (answer matches)
    assert 1.5 < r < 2.1

def test_r0_hallucination_still_scores_evidence_credit():
    # GRACE R0 does NOT penalize hallucination — a key weakness R1 fixes.
    resp = (
        "<evidence>Fabricated support text.</evidence>"
        "<answer>Christopher Nolan</answer>"
    )
    r, parts = r0_grace(resp, _sample_unans(), CFG)
    # Unanswerable + evidence path = wrong path for R0.
    assert r == 0.5  # only format bonus, no path reward
```

- [ ] **Step 6:** `pytest tests/test_r0.py -v`. Expected: all 6 FAIL with `ImportError`.

- [ ] **Step 7:** Implement `src/rewards/r0_grace.py` (replacing the skeleton version):

```python
"""R0 — GRACE baseline reward.

R0 = 0,                                    invalid format
     Rf,                                   valid format, wrong path
     Rf + Rp,                              correct <llm> path
     Rf + Rp + α·RLev + β·RLans,           correct <evidence> path
"""
from dataclasses import dataclass
from src.common.types import Sample
from src.common.parse import parse_response
from src.rewards.metrics import rouge_l

@dataclass(frozen=True)
class R0Config:
    Rf: float = 0.5
    Rp: float = 1.0
    alpha: float = 0.5
    beta: float = 0.5

def r0_grace(
    response: str, sample: Sample, cfg: R0Config = R0Config()
) -> tuple[float, dict]:
    parsed = parse_response(response)
    parts: dict = {
        "path": parsed.path,
        "format_valid": parsed.format_valid,
        "correct_path": False,
        "rouge_l_evidence": 0.0,
        "rouge_l_answer": 0.0,
    }

    if not parsed.format_valid:
        return 0.0, parts

    correct_path = (
        (parsed.path == "evidence" and sample.is_answerable)
        or (parsed.path == "llm" and not sample.is_answerable)
    )
    parts["correct_path"] = correct_path

    if not correct_path:
        return cfg.Rf, parts

    if parsed.path == "llm":
        return cfg.Rf + cfg.Rp, parts

    # Correct <evidence> path.
    rl_ev = rouge_l(" ".join(parsed.quoted_evidence), sample.gold_evidence)
    rl_ans = rouge_l(parsed.answer, sample.gold_answer)
    parts["rouge_l_evidence"] = rl_ev
    parts["rouge_l_answer"] = rl_ans
    return cfg.Rf + cfg.Rp + cfg.alpha * rl_ev + cfg.beta * rl_ans, parts
```

- [ ] **Step 8:** `pytest tests/test_r0.py -v`. Expected: all 6 PASS.

- [ ] **Step 9 (checkpoint):** Member 2's lane green for Week 1. Suggested commit: `feat(rewards): R0 GRACE reward + ROUGE-L metrics + full unit test coverage`

---

### Task 1.4 (Member 3, Days 4-7): Promote skeleton launcher + logging + results convention

**Files:**
- Modify: `scripts/launch_run.sh` (promote from `launch_skeleton.sh`)
- Create: `src/training/run_naming.py`
- Create: `tests/test_run_naming.py`
- Create: `docs/results-convention.md`

- [ ] **Step 1 (TDD):** Write failing test for run naming — `tests/test_run_naming.py`:

```python
from src.training.run_naming import make_run_id

def test_run_id_format():
    rid = make_run_id(reward="r0", optimizer="grpo", model="qwen3-0.6b", seed=0)
    # Expect: r0_grpo_qwen3-0.6b_s0_<timestamp>
    parts = rid.split("_")
    assert parts[0] == "r0"
    assert parts[1] == "grpo"
    assert parts[2] == "qwen3-0.6b"
    assert parts[3] == "s0"
    assert len(parts[4]) == 15  # YYYYMMDD-HHMMSS

def test_run_id_unique_across_calls():
    import time
    a = make_run_id("r0", "grpo", "qwen3-0.6b", 0)
    time.sleep(1.1)
    b = make_run_id("r0", "grpo", "qwen3-0.6b", 0)
    assert a != b
```

- [ ] **Step 2:** `pytest tests/test_run_naming.py -v`. Expected: `ImportError`.

- [ ] **Step 3:** Implement `src/training/run_naming.py`:

```python
"""Standardized run id. Every training run writes to results/<run_id>/."""
from datetime import datetime

def make_run_id(reward: str, optimizer: str, model: str, seed: int) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{reward}_{optimizer}_{model}_s{seed}_{ts}"
```

- [ ] **Step 4:** `pytest tests/test_run_naming.py -v`. Expected: PASS.

- [ ] **Step 5:** Promote `scripts/launch_skeleton.sh` to `scripts/launch_run.sh`:

```bash
#!/usr/bin/env bash
# Usage: launch_run.sh <config.yaml>
set -euo pipefail
CFG=${1:?"usage: launch_run.sh <config.yaml>"}
source ~/rl-rag-venv/bin/activate
cd $(dirname $0)/..

RUN_ID=$(python -c "
from src.training.run_naming import make_run_id
import yaml, sys
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
```

- [ ] **Step 6:** Add a `_meta` block to `configs/skeleton.yaml`:

```yaml
_meta:
  reward: r0
  optimizer: grpo
  model: qwen3-0.6b
  seed: 0
```

- [ ] **Step 7:** Create `docs/results-convention.md`:

```markdown
# Results convention

Every training run writes to `results/<run_id>/` where:

- `<run_id>` = `<reward>_<optimizer>_<model>_s<seed>_<YYYYMMDD-HHMMSS>`
- `config.yaml` — copy of the launched config
- `train.log` — stdout/stderr of the training command
- `checkpoints/` — model checkpoints (gitignored)
- `rollouts/` — sampled rollouts per step (gitignored, large)
- `metrics.json` — final evaluation metrics (added by Member 1's `evaluate.py`)
- `audit.json` — reward-hacking audit (added by Member 2's audit script)

Sync: see `docs/team.md` for the agreed mechanism.
```

- [ ] **Step 8:** Re-run the skeleton: `bash scripts/launch_run.sh configs/skeleton.yaml`. Expected: a new `results/r0_grpo_qwen3-0.6b_s0_<ts>/` directory exists with `config.yaml` and `train.log`.

- [ ] **Step 9 (checkpoint):** Member 3's lane green. Suggested commit: `feat(training): promote skeleton to real launch_run.sh with run_id + results convention`

---

### Week 1 Gate verification

Run on each laptop and the server:

- [ ] `pytest tests/ -v` — all tests from Tasks 1.2, 1.3, 1.4 pass on each member's laptop
- [ ] `bash scripts/launch_run.sh configs/skeleton.yaml` completes on the server end-to-end
- [ ] `src/common/types.Sample` imported in both `src/data/hotpot.py` and `tests/test_r0.py` — verify with `grep -r "from src.common.types" src tests`
- [ ] `src/common/parse.parse_response` imported in `src/rewards/r0_grace.py` — verify with `grep "from src.common.parse" src/rewards/*.py`
- [ ] `docs/team.md` records the results-sync mechanism

**If any item fails, do not start Week 2.** Pair-debug until green.

---

## Week 2 (Oct 19–25) — Full HotpotQA + R0 reproduction

**Week goal:** full-size R0 + GRPO run on HotpotQA produces Balanced Accuracy within 10 points of GRACE's published number (81% ± 10 → 71-91% acceptable).

### Task 2.1 (Member 1): Retrievers + balanced split + retrieval cache

**Files:**
- Create: `src/data/retrievers.py`
- Create: `src/data/balanced_split.py`
- Create: `src/data/cache.py`
- Create: `scripts/download_hotpot_full.py`
- Create: `scripts/build_retrieval.py`
- Create: `tests/test_retrievers.py`
- Create: `tests/test_balanced_split.py`

- [ ] **Step 1 (TDD):** Write failing test for retrievers — `tests/test_retrievers.py`:

```python
from src.data.retrievers import BM25Retriever, DenseRetriever

CORPUS = [
    "Christopher Nolan directed Inception in 2010.",
    "Stanley Kubrick directed 2001: A Space Odyssey.",
    "Steven Spielberg directed Jaws in 1975.",
    "Quentin Tarantino directed Pulp Fiction.",
    "Martin Scorsese directed Goodfellas.",
]

def test_bm25_returns_top_k_in_order():
    r = BM25Retriever(CORPUS)
    hits = r.retrieve("Who directed Inception?", k=2)
    assert len(hits) == 2
    assert "Nolan" in hits[0]

def test_dense_returns_top_k():
    r = DenseRetriever(CORPUS, model_name="Qwen/Qwen3-Embedding-0.6B")
    hits = r.retrieve("Who directed Inception?", k=2)
    assert len(hits) == 2
    assert "Nolan" in hits[0]
```

- [ ] **Step 2:** `pytest tests/test_retrievers.py -v`. Expected: `ImportError`.

- [ ] **Step 3:** Implement `src/data/retrievers.py`:

```python
"""BM25 and dense retrievers. Both expose the same `retrieve(query, k)` interface."""
from typing import Protocol
import numpy as np
from rank_bm25 import BM25Okapi

class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[str]: ...

def _tokenize(text: str) -> list[str]:
    return text.lower().split()

class BM25Retriever:
    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        self._bm25 = BM25Okapi([_tokenize(c) for c in corpus])

    def retrieve(self, query: str, k: int) -> list[str]:
        scores = self._bm25.get_scores(_tokenize(query))
        idx = np.argsort(scores)[::-1][:k]
        return [self.corpus[i] for i in idx]

class DenseRetriever:
    def __init__(self, corpus: list[str], model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        from sentence_transformers import SentenceTransformer
        self.corpus = corpus
        self._model = SentenceTransformer(model_name)
        self._embs = self._model.encode(
            corpus, convert_to_numpy=True, normalize_embeddings=True
        )

    def retrieve(self, query: str, k: int) -> list[str]:
        q_emb = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = self._embs @ q_emb
        idx = np.argsort(sims)[::-1][:k]
        return [self.corpus[i] for i in idx]
```

- [ ] **Step 4:** `pytest tests/test_retrievers.py -v`. Expected: PASS (dense test is slow due to model download — mark it with `@pytest.mark.slow` if you want to skip in CI).

- [ ] **Step 5 (TDD):** Write failing test for balanced split — `tests/test_balanced_split.py`:

```python
from src.common.types import Sample
from src.data.balanced_split import build_balanced_split

def _mk(sid, is_ans):
    return Sample(
        sample_id=sid, question="q",
        gold_passages=["gold"] if is_ans else [],
        distractor_passages=["d1", "d2", "d3", "d4", "d5"],
        gold_answer="a", gold_evidence=["gold"],
        retrieved_passages=[], is_answerable=True,
    )

def test_balanced_split_marks_unanswerable_when_gold_missing_from_retrieved():
    # Fake retriever always returns distractors only.
    samples = [_mk(f"s{i}", True) for i in range(10)]
    def fake_retrieve(q, k):
        return samples[0].distractor_passages[:k]
    out = build_balanced_split(samples, retrieve_fn=fake_retrieve, k=5)
    assert all(s.is_answerable is False for s in out)
    assert all(len(s.retrieved_passages) == 5 for s in out)

def test_balanced_split_marks_answerable_when_gold_in_retrieved():
    samples = [_mk(f"s{i}", True) for i in range(10)]
    def fake_retrieve(q, k):
        return samples[0].gold_passages + samples[0].distractor_passages[:k-1]
    out = build_balanced_split(samples, retrieve_fn=fake_retrieve, k=5)
    assert all(s.is_answerable is True for s in out)
```

- [ ] **Step 6:** `pytest tests/test_balanced_split.py -v`. Expected: `ImportError`.

- [ ] **Step 7:** Implement `src/data/balanced_split.py`:

```python
"""Build answerable/unanswerable split based on retriever output.

For each sample: retrieve k passages. Mark `is_answerable = True` if any
gold passage is in the retrieved set, else `False`. This matches the GRACE
definition and the ans:unans ratios reported in the proposal (296:204 for
HotpotQA with dense retriever at k=5).
"""
from typing import Callable
from src.common.types import Sample

RetrieveFn = Callable[[str, int], list[str]]

def build_balanced_split(
    samples: list[Sample], retrieve_fn: RetrieveFn, k: int = 5
) -> list[Sample]:
    out = []
    for s in samples:
        retrieved = retrieve_fn(s.question, k)
        is_ans = any(gp in retrieved for gp in s.gold_passages)
        out.append(
            Sample(
                sample_id=s.sample_id,
                question=s.question,
                gold_passages=s.gold_passages,
                distractor_passages=s.distractor_passages,
                gold_answer=s.gold_answer,
                gold_evidence=s.gold_evidence,
                retrieved_passages=retrieved,
                is_answerable=is_ans,
            )
        )
    return out
```

- [ ] **Step 8:** `pytest tests/test_balanced_split.py -v`. Expected: both PASS.

- [ ] **Step 9:** Implement `src/data/cache.py`:

```python
"""Disk cache for retrieval results. Key by (retriever_name, question, k)."""
from pathlib import Path
import json
import hashlib

def _key(retriever_name: str, question: str, k: int) -> str:
    h = hashlib.sha256(f"{retriever_name}|{question}|{k}".encode()).hexdigest()[:16]
    return h

class RetrievalCache:
    def __init__(self, dir: Path):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, retriever_name: str, question: str, k: int) -> list[str] | None:
        p = self.dir / f"{_key(retriever_name, question, k)}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def set(self, retriever_name: str, question: str, k: int, passages: list[str]) -> None:
        p = self.dir / f"{_key(retriever_name, question, k)}.json"
        p.write_text(json.dumps(passages), encoding="utf-8")
```

- [ ] **Step 10:** Write `scripts/download_hotpot_full.py`:

```python
"""Download full HotpotQA train (2000) + validation (500) subsets per proposal."""
from datasets import load_dataset
from pathlib import Path
import json

for split, n, out in [
    ("train", 2000, Path("data/raw/hotpot_train.jsonl")),
    ("validation", 500, Path("data/raw/hotpot_val.jsonl")),
]:
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("hotpot_qa", "distractor", split=f"{split}[:{n}]")
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")
    print(f"{out}: {n} samples")
```

Run it: `python scripts/download_hotpot_full.py`.

- [ ] **Step 11:** Write `scripts/build_retrieval.py` that builds the balanced split with both retrievers and writes to `data/cache/<retriever>_<split>.jsonl`:

```python
"""Build retrieval + balanced split for train/val with BM25 and dense retrievers.

Writes one file per (retriever, split) in data/cache/.
"""
from pathlib import Path
import json
from src.data.hotpot import load_hotpot_tiny  # also works for full files
from src.data.retrievers import BM25Retriever, DenseRetriever
from src.data.balanced_split import build_balanced_split
from src.data.cache import RetrievalCache

K = 5
SPLITS = [
    ("train", Path("data/raw/hotpot_train.jsonl")),
    ("val",   Path("data/raw/hotpot_val.jsonl")),
]

def all_passages(samples) -> list[str]:
    seen, out = set(), []
    for s in samples:
        for p in s.gold_passages + s.distractor_passages:
            if p not in seen:
                seen.add(p); out.append(p)
    return out

def dump(samples, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.__dict__) + "\n")

for split, raw in SPLITS:
    samples = load_hotpot_tiny(raw)  # name is misleading — works on full too
    corpus = all_passages(samples)
    for name, R in [("bm25", BM25Retriever(corpus)),
                    ("dense", DenseRetriever(corpus))]:
        split_samples = build_balanced_split(samples, R.retrieve, k=K)
        n_ans = sum(s.is_answerable for s in split_samples)
        print(f"{name}/{split}: n={len(split_samples)}, ans={n_ans}, unans={len(split_samples)-n_ans}")
        dump(split_samples, Path(f"data/cache/{name}_{split}.jsonl"))
```

- [ ] **Step 12:** Run `python scripts/build_retrieval.py` on the server. Expected: 4 files in `data/cache/`, console shows ans:unans ratios reasonable (dense/val should be close to 296:204 per proposal).

- [ ] **Step 13:** Write `docs/data.md` documenting the ratios actually observed. If dense/val ans:unans is wildly off from 296:204, debug before moving on (likely cause: gold-passage identity check needs fuzzy matching rather than exact string equality — switch to title-based matching if so).

- [ ] **Step 14 (checkpoint):** Member 1's Week 2 lane green. Suggested commit: `feat(data): retrievers + balanced split + cache for full HotpotQA`

---

### Task 2.2 (Member 2): Harden R0 with edge-case tests + sketch R1 design

**Files:**
- Modify: `tests/test_r0.py` (add edge cases)
- Create: `docs/rewards.md`

- [ ] **Step 1:** Add edge-case tests to `tests/test_r0.py`:

```python
def test_r0_question_copy_still_scores_evidence_credit():
    # If the model copies the question into the answer, R0 doesn't notice.
    # Documented weakness — audit will catch this at eval time.
    s = _sample_ans(gold_answer="Christopher Nolan")
    resp = (
        "<evidence>Christopher Nolan directed Inception.</evidence>"
        "<answer>Who directed Inception?</answer>"
    )
    r, parts = r0_grace(resp, s, CFG)
    # Low RLans but still gets Rf + Rp + α·high
    assert r > 1.5

def test_r0_zero_overlap_evidence():
    s = _sample_ans()
    resp = (
        "<evidence>Completely unrelated text about dolphins.</evidence>"
        "<answer>Christopher Nolan</answer>"
    )
    r, parts = r0_grace(resp, s, CFG)
    assert parts["rouge_l_evidence"] < 0.2
    assert parts["rouge_l_answer"] == pytest.approx(1.0)

def test_r0_llm_path_on_answerable_still_wrong_path():
    s = _sample_ans()
    resp = "<llm>cannot answer</llm><answer>unknown</answer>"
    r, parts = r0_grace(resp, s, CFG)
    assert r == 0.5
    assert parts["correct_path"] is False
```

(Add `import pytest` at the top of the test file.)

- [ ] **Step 2:** `pytest tests/test_r0.py -v`. Expected: all R0 tests PASS (9 total now).

- [ ] **Step 3:** Write `docs/rewards.md` documenting the R0 → R1 → R2 progression and locking in R1's formula:

```markdown
# Reward design

See spec §4 for background. This file is the implementation-level reference.

## R0 — GRACE baseline

Implemented in `src/rewards/r0_grace.py`. Config: `R0Config(Rf=0.5, Rp=1.0, alpha=0.5, beta=0.5)`.

## R1 — Cost-sensitive (to be implemented Week 3)

```
R1 = -λ,                                              hallucinated (answered on unanswerable)
      0,                                              invalid format
      Rf,                                             valid format, wrong path (non-hallucination)
      Rf + Rp + γ·RLans,                              correct <llm> path
      Rf·𝟙[quote ⊂ Cret] + Rp + α·RLev + β·RLans,   correct <evidence> path
```

Config: `R1Config(Rf=0.5, Rp=1.0, alpha=0.5, beta=0.5, gamma=0.5, lam=1.0)`.

"Hallucinated" = `parsed.path == "evidence" and not sample.is_answerable`.
"Quote ⊂ Cret" = every quoted evidence string is a substring of at least one `sample.retrieved_passages` entry.

## R2 — Hard-gated multiplicative (to be implemented Week 3)

```
R2 = Rformat × (w_p·Rpath + w_q · Rpath · Rquality)
```

Config: `R2Config(w_p=0.4, w_q=0.6)`.

- `Rformat ∈ {0,1}` — valid tags AND (if evidence path) every quote is a substring of retrieved passages.
- `Rpath ∈ {0,1}` — correct <evidence>/<llm> choice given `is_answerable`.
- `Rquality ∈ [0,1]`:
  - evidence path: `0.5·RLev + 0.5·RLans`
  - llm path:      `RLans` (quality of disclaimer vs reference)

## Reward-hacking audit (Week 3)

Per-batch stats computed by `src/rewards/audit.py`:
- share of responses on each path (evidence / llm / invalid)
- mean + histogram of quoted-evidence length (in tokens)
- fraction of answers that contain the question verbatim
- fraction of evidence quotes NOT found in `retrieved_passages`
```

- [ ] **Step 4 (checkpoint):** Suggested commit: `test(rewards): edge-case coverage for R0 + R1/R2 design doc`

---

### Task 2.3 (Member 3): Full-size GRPO config + eval harness skeleton

**Files:**
- Create: `configs/r0_grpo.yaml`
- Create: `src/eval/metrics.py`
- Create: `src/eval/evaluate.py`
- Create: `tests/test_eval_metrics.py`

- [ ] **Step 1 (TDD): Write failing test for eval metrics** — `tests/test_eval_metrics.py`:

```python
from src.eval.metrics import exact_match, abstention_accuracy, balanced_accuracy

def test_em_exact():
    assert exact_match("Christopher Nolan", "Christopher Nolan") == 1.0

def test_em_case_insensitive_and_punct():
    assert exact_match("christopher  nolan.", "Christopher Nolan") == 1.0

def test_em_mismatch():
    assert exact_match("Kubrick", "Christopher Nolan") == 0.0

def test_abstention_accuracy_correct_refusals():
    # (path, is_answerable) tuples
    preds = [("llm", False), ("evidence", True), ("llm", False), ("evidence", False)]
    # True negatives: unanswerable samples routed to <llm>.
    # Denominator: all unanswerable samples.
    assert abstention_accuracy(preds) == 2 / 3

def test_balanced_accuracy_simple():
    # 50% EM on answerable, 100% abstention on unanswerable → 0.75
    assert balanced_accuracy(em_answerable=0.5, abstention_unanswerable=1.0) == 0.75
```

- [ ] **Step 2:** `pytest tests/test_eval_metrics.py -v`. Expected: `ImportError`.

- [ ] **Step 3:** Implement `src/eval/metrics.py`:

```python
"""Evaluation metrics. Separate from reward metrics (reward uses ROUGE-L shaping;
eval uses EM/BERTScore/LJ for comparison against baselines)."""
import re
import string

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[" + re.escape(string.punctuation) + r"]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0

def abstention_accuracy(predictions: list[tuple[str, bool]]) -> float:
    """predictions: list of (path, is_answerable). Returns TN rate over unanswerable."""
    unans = [p for p in predictions if not p[1]]
    if not unans:
        return 0.0
    correct = sum(1 for p, _ in unans if p == "llm")
    return correct / len(unans)

def balanced_accuracy(em_answerable: float, abstention_unanswerable: float) -> float:
    return 0.5 * em_answerable + 0.5 * abstention_unanswerable
```

- [ ] **Step 4:** `pytest tests/test_eval_metrics.py -v`. Expected: all 5 PASS.

- [ ] **Step 5:** Write `src/eval/evaluate.py`:

```python
"""CLI: evaluate a checkpoint on a split.

Usage:
    python -m src.eval.evaluate --checkpoint path/to/ckpt \
        --split data/cache/dense_val.jsonl \
        --out results/<run_id>/metrics.json
"""
import argparse, json
from pathlib import Path
from statistics import mean

from src.common.types import Sample
from src.common.parse import parse_response
from src.prompts.templates import build_prompt
from src.eval.metrics import exact_match, abstention_accuracy, balanced_accuracy

def _load_split(path: Path) -> list[Sample]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out.append(Sample(**row))
    return out

def _generate(model, tokenizer, prompts: list[str]) -> list[str]:
    # Server-side: use vLLM for throughput. Laptop fallback: HF generate.
    try:
        from vllm import LLM, SamplingParams
        if not isinstance(model, LLM):
            model = LLM(model=model, dtype="float16", gpu_memory_utilization=0.6)
        params = SamplingParams(temperature=0.0, max_tokens=512)
        outs = model.generate(prompts, params)
        return [o.outputs[0].text for o in outs]
    except Exception:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if isinstance(model, str):
            tokenizer = AutoTokenizer.from_pretrained(model)
            model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16).cuda()
        results = []
        for p in prompts:
            ids = tokenizer(p, return_tensors="pt").to(model.device)
            out = model.generate(**ids, max_new_tokens=512, do_sample=False)
            results.append(tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
        return results

def evaluate(checkpoint: str, split_file: Path, out: Path) -> dict:
    samples = _load_split(split_file)
    prompts = [build_prompt(s.question, s.retrieved_passages) for s in samples]
    responses = _generate(checkpoint, None, prompts)

    rows = []
    for s, resp in zip(samples, responses):
        parsed = parse_response(resp)
        em = exact_match(parsed.answer, s.gold_answer) if s.is_answerable else 0.0
        rows.append({
            "sample_id": s.sample_id,
            "is_answerable": s.is_answerable,
            "path": parsed.path,
            "answer": parsed.answer,
            "em": em,
        })

    ans_rows = [r for r in rows if r["is_answerable"]]
    em_ans = mean(r["em"] for r in ans_rows) if ans_rows else 0.0
    abst = abstention_accuracy([(r["path"], r["is_answerable"]) for r in rows])
    bal = balanced_accuracy(em_ans, abst)

    metrics = {
        "n": len(rows),
        "n_answerable": len(ans_rows),
        "em_answerable": em_ans,
        "abstention_unanswerable": abst,
        "balanced_accuracy": bal,
        "path_shares": {
            "evidence": sum(1 for r in rows if r["path"] == "evidence") / len(rows),
            "llm":      sum(1 for r in rows if r["path"] == "llm") / len(rows),
            "invalid":  sum(1 for r in rows if r["path"] == "invalid") / len(rows),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Also dump per-row predictions for later analysis.
    (out.parent / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    return metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    m = evaluate(args.checkpoint, args.split, args.out)
    print(json.dumps(m, indent=2))
```

(BERTScore and LJ are added in Week 3 by Member 1 — this is the skeleton.)

- [ ] **Step 6:** Write `configs/r0_grpo.yaml` (full-size GRPO config). Base on the GRACE repo's working GRPO YAML, swap paths to our files:

```yaml
_meta:
  reward: r0
  optimizer: grpo
  model: qwen3-0.6b
  seed: 0

trainer:
  project_name: rl-rag
  experiment_name: r0_grpo_qwen3-0.6b
  n_gpus_per_node: 1
  nnodes: 1
  total_training_steps: 400   # tune after first run
  save_freq: 100
  logger: wandb   # or ['console']

data:
  train_files: data/cache/dense_train.jsonl
  val_files:   data/cache/dense_val.jsonl
  prompt_key: prompt
  max_prompt_length: 2048
  max_response_length: 512
  train_batch_size: 32

actor_rollout_ref:
  model:
    path: Qwen/Qwen3-0.6B
    enable_lora: true
    lora_rank: 16
    lora_alpha: 32
  rollout:
    name: vllm
    gpu_memory_utilization: 0.6
    n: 8  # G rollouts for GRPO
    temperature: 0.9
  actor:
    optim:
      lr: 1e-5
    ppo_mini_batch_size: 16

algorithm:
  name: grpo
  kl_coef: 0.001
  advantage_normalization: true

reward_fn:
  path: src.rewards.r0_grace:r0_grace
```

(Note: the YAML field names here are illustrative. On first run, verl will likely complain about mismatches — fix each one against verl's actual schema in `verl/trainer/config/`.)

- [ ] **Step 7:** Make the dataset loader for verl output a prompt column. The simplest approach: have `scripts/build_retrieval.py` additionally write a `prompt` field by calling `build_prompt(question, retrieved_passages)`. Add this to that script:

```python
# in scripts/build_retrieval.py dump() function, modify:
def dump(samples, path: Path):
    from src.prompts.templates import build_prompt
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            row = s.__dict__.copy()
            row["prompt"] = build_prompt(s.question, s.retrieved_passages)
            f.write(json.dumps(row) + "\n")
```

Then re-run `python scripts/build_retrieval.py`.

- [ ] **Step 8:** On the server, launch:

```bash
bash scripts/launch_run.sh configs/r0_grpo.yaml
```

Expect ~2-6 hours for 400 steps. Monitor `train.log` and W&B/tensorboard. First launches typically fail on config schema — iterate on the YAML.

- [ ] **Step 9:** When training completes, evaluate:

```bash
python -m src.eval.evaluate \
  --checkpoint results/r0_grpo_qwen3-0.6b_s0_*/checkpoints/global_step_400 \
  --split data/cache/dense_val.jsonl \
  --out results/r0_grpo_qwen3-0.6b_s0_*/metrics.json
```

- [ ] **Step 10 (Week 2 Gate check):** Read `metrics.json`. **Balanced Accuracy should be in [0.71, 0.91]** (10 points of GRACE's 0.81). If not — **halt the whole team**, pair-debug. Common issues: prompt format drift from GRACE, wrong LoRA rank, learning rate too high, distractor-swap not matching GRACE's definition.

- [ ] **Step 11 (checkpoint):** Suggested commit: `feat(training,eval): first full R0+GRPO run on HotpotQA with evaluate.py`

---

### Week 2 Gate verification

- [ ] `data/cache/{bm25,dense}_{train,val}.jsonl` all exist
- [ ] `docs/data.md` documents the observed ans:unans ratios (dense/val target: ~296:204)
- [ ] `pytest tests/ -v` all pass
- [ ] `metrics.json` for the first full R0+GRPO run exists
- [ ] Balanced Accuracy ∈ [0.71, 0.91]

---

## Week 3 (Oct 26–Nov 1) — Reward variants + Groq LJ + DAPO

**Week goal:** R1 + R2 implemented and tested; DAPO config working; LJ metric integrated.

### Task 3.1 (Member 1): Groq LJ with on-disk cache + LJ validation set

**Files:**
- Create: `src/eval/judge.py`
- Create: `tests/test_judge.py`
- Create: `data/lj_validation/README.md`
- Create: `data/lj_validation/labels.jsonl`
- Modify: `src/eval/evaluate.py` (add LJ, BERTScore)

- [ ] **Step 1 (TDD):** Failing tests — `tests/test_judge.py`:

```python
from unittest.mock import patch, MagicMock
from src.eval.judge import GroqJudge

def test_judge_calls_api_and_returns_score():
    judge = GroqJudge(api_key="fake", cache_dir=None)
    fake_response = MagicMock()
    fake_response.json.return_value = {"choices": [{"message": {"content": "SCORE: 1"}}]}
    fake_response.status_code = 200
    with patch("src.eval.judge.requests.post", return_value=fake_response) as post:
        s = judge.score("What is 2+2?", "4", "4")
    assert s == 1.0
    assert post.called

def test_judge_parses_score_from_messy_output():
    judge = GroqJudge(api_key="fake", cache_dir=None)
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "The answer is semantically equivalent. SCORE: 1"}}]
    }
    fake_response.status_code = 200
    with patch("src.eval.judge.requests.post", return_value=fake_response):
        s = judge.score("q", "ans", "ref")
    assert s == 1.0

def test_judge_cache_hit_skips_api(tmp_path):
    judge = GroqJudge(api_key="fake", cache_dir=tmp_path)
    # Prime the cache
    judge._cache_set("key1", 1.0)
    with patch("src.eval.judge.requests.post") as post:
        s = judge._cache_get("key1")
    assert s == 1.0
    assert not post.called
```

- [ ] **Step 2:** `pytest tests/test_judge.py -v`. Expected: `ImportError`.

- [ ] **Step 3:** Implement `src/eval/judge.py`:

```python
"""Groq LLM-as-Judge client with on-disk cache. Uses Llama 3.3 70B at temp=0."""
import os, json, hashlib, re
from pathlib import Path
import requests
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

MODEL = "llama-3.3-70b-versatile"
URL = "https://api.groq.com/openai/v1/chat/completions"

PROMPT = """You are a strict grader. Given a question, a predicted answer, and a reference answer, decide if the prediction is semantically equivalent to the reference.

Question: {q}
Prediction: {pred}
Reference: {ref}

Respond with exactly: SCORE: 1 (if equivalent) or SCORE: 0 (if not). No other text."""

def _hash(parts) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()[:20]

class GroqJudge:
    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None):
        self.api_key = api_key or os.environ["GROQ_API_KEY"]
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / f"{key}.json" if self.cache_dir else None

    def _cache_get(self, key: str) -> float | None:
        p = self._cache_path(key)
        if p and p.exists():
            return json.loads(p.read_text())["score"]
        return None

    def _cache_set(self, key: str, score: float):
        p = self._cache_path(key)
        if p:
            p.write_text(json.dumps({"score": score}))

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(requests.HTTPError),
    )
    def _call(self, prompt: str) -> str:
        resp = requests.post(
            URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 10,
            },
            timeout=30,
        )
        if resp.status_code == 429:
            raise requests.HTTPError("rate limit")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def score(self, question: str, prediction: str, reference: str) -> float:
        key = _hash([MODEL, question, prediction, reference])
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        text = self._call(PROMPT.format(q=question, pred=prediction, ref=reference))
        m = re.search(r"SCORE:\s*([01])", text)
        score = float(m.group(1)) if m else 0.0
        self._cache_set(key, score)
        return score
```

- [ ] **Step 4:** `pytest tests/test_judge.py -v`. Expected: all 3 PASS.

- [ ] **Step 5:** Create the 50-example LJ validation set. Open `data/cache/dense_val.jsonl`, pick 50 answerable samples, run the baseline Qwen3-0.6B zero-shot through them, hand-label each (prediction_equivalent_to_gold ∈ {0, 1}):

Write a helper `scripts/make_lj_validation.py`:

```python
"""Sample 50 answerable rows and write labels.jsonl stub for hand-labeling."""
import json
from pathlib import Path
import random

random.seed(42)
rows = [json.loads(l) for l in Path("data/cache/dense_val.jsonl").open(encoding="utf-8")]
ans = [r for r in rows if r["is_answerable"]][:200]
pick = random.sample(ans, 50)

OUT = Path("data/lj_validation/labels.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for r in pick:
        f.write(json.dumps({
            "sample_id": r["sample_id"],
            "question": r["question"],
            "gold_answer": r["gold_answer"],
            "prediction": "",   # fill after baseline inference
            "em": None,          # fill via src.eval.metrics.exact_match
            "lj_hand": None,     # 0 or 1, human-labeled
        }) + "\n")
print(f"Wrote {OUT}. Now: run baseline on these 50, fill prediction/em/lj_hand.")
```

Run the baseline on these 50 (reuse `src.eval.evaluate._generate`), fill `prediction` and `em`. Then hand-label `lj_hand`. Document the process in `data/lj_validation/README.md`.

- [ ] **Step 6:** Validation gate — write `scripts/validate_lj.py`:

```python
"""Check that LJ score agrees with hand labels at ≥85% on the validation set."""
import json
from pathlib import Path
from src.eval.judge import GroqJudge

rows = [json.loads(l) for l in Path("data/lj_validation/labels.jsonl").open(encoding="utf-8")]
judge = GroqJudge(cache_dir=Path("data/lj_validation/cache"))

agree = 0
for r in rows:
    lj = judge.score(r["question"], r["prediction"], r["gold_answer"])
    if lj == r["lj_hand"]:
        agree += 1
print(f"LJ agreement with hand labels: {agree}/{len(rows)} = {agree/len(rows):.2%}")
assert agree / len(rows) >= 0.85, "LJ sanity check failed."
```

Run it. **If agreement < 85%, revise the LJ prompt in `src/eval/judge.py` and re-run.**

- [ ] **Step 7:** Extend `src/eval/evaluate.py` to include BERTScore + LJ:

```python
# At the top, add:
from bert_score import score as bert_score
from src.eval.judge import GroqJudge

# Inside evaluate(), after computing rows:
preds = [r["answer"] for r in rows if r["is_answerable"]]
refs = [s.gold_answer for s, r in zip(samples, rows) if r["is_answerable"]]
if preds:
    _, _, bs_f1 = bert_score(preds, refs, lang="en", rescale_with_baseline=True)
    bert_f1_mean = float(bs_f1.mean())
else:
    bert_f1_mean = 0.0

judge = GroqJudge(cache_dir=Path("data/lj_cache"))
lj_scores = []
for s, r in zip(samples, rows):
    if r["is_answerable"]:
        lj_scores.append(judge.score(s.question, r["answer"], s.gold_answer))
lj_mean = sum(lj_scores) / len(lj_scores) if lj_scores else 0.0

# Add to metrics dict:
metrics["bert_f1_answerable"] = bert_f1_mean
metrics["lj_answerable"] = lj_mean
```

- [ ] **Step 8:** Re-run eval on the Week 2 R0 checkpoint, this time with LJ + BERTScore enabled. Spot-check `metrics.json` — all four metrics present and non-zero.

- [ ] **Step 9 (checkpoint):** Suggested commit: `feat(eval): Groq LJ with cache + BERTScore + 50-example LJ validation`

---

### Task 3.2 (Member 2): Implement R1 and R2 with full test coverage + audit utilities

**Files:**
- Create: `src/rewards/r1_cost_sensitive.py`
- Create: `src/rewards/r2_hard_gated.py`
- Create: `src/rewards/audit.py`
- Create: `tests/test_r1.py`
- Create: `tests/test_r2.py`
- Create: `tests/test_audit.py`

- [ ] **Step 1 (TDD):** Failing tests for R1 — `tests/test_r1.py`:

```python
import pytest
from src.common.types import Sample
from src.rewards.r1_cost_sensitive import r1_cost_sensitive, R1Config

CFG = R1Config()  # defaults from docs/rewards.md

def _ans(**k):
    base = dict(
        sample_id="s", question="Who directed Inception?",
        gold_passages=["Nolan directed Inception."],
        distractor_passages=["x"],
        gold_answer="Christopher Nolan",
        gold_evidence=["Nolan directed Inception."],
        retrieved_passages=["Nolan directed Inception.", "x"],
        is_answerable=True,
    ); base.update(k); return Sample(**base)

def _unans(**k):
    return _ans(gold_passages=[], is_answerable=False,
                retrieved_passages=["x", "y", "z"], **k)

def test_r1_hallucination_scores_negative_lambda():
    # Answered on unanswerable → hallucination.
    resp = "<evidence>Fabricated.</evidence><answer>Christopher Nolan</answer>"
    r, _ = r1_cost_sensitive(resp, _unans(), CFG)
    assert r == -CFG.lam

def test_r1_invalid_format_scores_zero():
    r, _ = r1_cost_sensitive("junk", _ans(), CFG)
    assert r == 0.0

def test_r1_wrong_path_nonhallucination_scores_rf():
    # Answerable, model abstains. Not a hallucination — wrong path only.
    resp = "<llm>no</llm><answer>unknown</answer>"
    r, _ = r1_cost_sensitive(resp, _ans(), CFG)
    assert r == CFG.Rf

def test_r1_correct_llm_path_includes_disclaimer_quality():
    resp = "<llm>I cannot answer from the provided evidence.</llm><answer>unknown</answer>"
    r, parts = r1_cost_sensitive(resp, _unans(), CFG)
    assert r >= CFG.Rf + CFG.Rp  # at least floor
    assert "rouge_l_answer" in parts

def test_r1_evidence_path_quote_in_retrieved_full_credit():
    resp = "<evidence>Nolan directed Inception.</evidence><answer>Christopher Nolan</answer>"
    r, parts = r1_cost_sensitive(resp, _ans(), CFG)
    assert parts["quote_grounded"] is True
    assert r > CFG.Rf + CFG.Rp

def test_r1_evidence_path_fabricated_quote_loses_rf_gate():
    # Quote not in retrieved passages → Rf gate drops to 0.
    resp = "<evidence>This string is not in the passages.</evidence><answer>Christopher Nolan</answer>"
    r, parts = r1_cost_sensitive(resp, _ans(), CFG)
    assert parts["quote_grounded"] is False
    # No Rf; still gets Rp + α·low_rouge + β·high_rouge
    assert 0 < r < CFG.Rf + CFG.Rp + CFG.alpha + CFG.beta
```

- [ ] **Step 2:** `pytest tests/test_r1.py -v`. Expected: all FAIL with `ImportError`.

- [ ] **Step 3:** Implement `src/rewards/r1_cost_sensitive.py`:

```python
"""R1 — Cost-sensitive reward with hallucination penalty + quote-containment gate."""
from dataclasses import dataclass
from src.common.types import Sample
from src.common.parse import parse_response
from src.rewards.metrics import rouge_l, quote_in_passages

@dataclass(frozen=True)
class R1Config:
    Rf: float = 0.5
    Rp: float = 1.0
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.5
    lam: float = 1.0

def r1_cost_sensitive(
    response: str, sample: Sample, cfg: R1Config = R1Config()
) -> tuple[float, dict]:
    parsed = parse_response(response)
    parts: dict = {
        "path": parsed.path,
        "format_valid": parsed.format_valid,
        "correct_path": False,
        "hallucination": False,
        "quote_grounded": False,
        "rouge_l_evidence": 0.0,
        "rouge_l_answer": 0.0,
    }

    # Hallucination = answered (evidence path) on unanswerable sample.
    if parsed.path == "evidence" and not sample.is_answerable:
        parts["hallucination"] = True
        return -cfg.lam, parts

    if not parsed.format_valid:
        return 0.0, parts

    correct_path = (
        (parsed.path == "evidence" and sample.is_answerable)
        or (parsed.path == "llm" and not sample.is_answerable)
    )
    parts["correct_path"] = correct_path

    if not correct_path:
        return cfg.Rf, parts

    if parsed.path == "llm":
        rl_ans = rouge_l(parsed.answer, sample.gold_answer)
        parts["rouge_l_answer"] = rl_ans
        return cfg.Rf + cfg.Rp + cfg.gamma * rl_ans, parts

    # Correct <evidence> path with quote-containment gate on Rf.
    grounded = all(quote_in_passages(q, sample.retrieved_passages) for q in parsed.quoted_evidence)
    parts["quote_grounded"] = grounded
    rl_ev = rouge_l(" ".join(parsed.quoted_evidence), sample.gold_evidence)
    rl_ans = rouge_l(parsed.answer, sample.gold_answer)
    parts["rouge_l_evidence"] = rl_ev
    parts["rouge_l_answer"] = rl_ans
    rf_gated = cfg.Rf if grounded else 0.0
    return rf_gated + cfg.Rp + cfg.alpha * rl_ev + cfg.beta * rl_ans, parts
```

- [ ] **Step 4:** `pytest tests/test_r1.py -v`. Expected: all PASS.

- [ ] **Step 5 (TDD):** Failing tests for R2 — `tests/test_r2.py`:

```python
import pytest
from src.common.types import Sample
from src.rewards.r2_hard_gated import r2_hard_gated, R2Config

CFG = R2Config()

def _ans(**k):
    base = dict(
        sample_id="s", question="q",
        gold_passages=["Nolan directed Inception."],
        distractor_passages=["x"],
        gold_answer="Christopher Nolan",
        gold_evidence=["Nolan directed Inception."],
        retrieved_passages=["Nolan directed Inception.", "x"],
        is_answerable=True,
    ); base.update(k); return Sample(**base)

def _unans(**k):
    return _ans(gold_passages=[], is_answerable=False,
                retrieved_passages=["x", "y", "z"], **k)

def test_r2_invalid_format_collapses_to_zero():
    r, _ = r2_hard_gated("junk", _ans(), CFG)
    assert r == 0.0

def test_r2_wrong_path_scores_zero_despite_valid_format():
    # Rpath=0 means both w_p*Rpath and w_q*Rpath*Rquality = 0.
    resp = "<llm>no</llm><answer>unknown</answer>"
    r, parts = r2_hard_gated(resp, _ans(), CFG)
    assert r == 0.0
    assert parts["R_format"] == 1
    assert parts["R_path"] == 0

def test_r2_fabricated_quote_collapses_to_zero():
    # Rformat gate also requires every quote ⊂ retrieved.
    resp = "<evidence>Not in passages.</evidence><answer>Christopher Nolan</answer>"
    r, parts = r2_hard_gated(resp, _ans(), CFG)
    assert r == 0.0
    assert parts["R_format"] == 0

def test_r2_correct_evidence_full_credit():
    resp = "<evidence>Nolan directed Inception.</evidence><answer>Christopher Nolan</answer>"
    r, parts = r2_hard_gated(resp, _ans(), CFG)
    # Rformat=1, Rpath=1, Rquality=0.5*1 + 0.5*1 = 1 → r = w_p + w_q = 1.0
    assert r == pytest.approx(CFG.w_p + CFG.w_q)

def test_r2_correct_llm_scores_wp_plus_wq_times_quality():
    resp = "<llm>I cannot answer.</llm><answer>unknown</answer>"
    r, parts = r2_hard_gated(resp, _unans(), CFG)
    assert r >= CFG.w_p  # at least Rpath credit, possibly more from disclaimer quality
```

- [ ] **Step 6:** `pytest tests/test_r2.py -v`. Expected: all FAIL with `ImportError`.

- [ ] **Step 7:** Implement `src/rewards/r2_hard_gated.py`:

```python
"""R2 — Hard-gated multiplicative reward. The RQ1 extreme test."""
from dataclasses import dataclass
from src.common.types import Sample
from src.common.parse import parse_response
from src.rewards.metrics import rouge_l, quote_in_passages

@dataclass(frozen=True)
class R2Config:
    w_p: float = 0.4
    w_q: float = 0.6

def r2_hard_gated(
    response: str, sample: Sample, cfg: R2Config = R2Config()
) -> tuple[float, dict]:
    parsed = parse_response(response)

    # R_format gate: valid tags AND (if evidence path) every quote in retrieved.
    r_format = 1
    if not parsed.format_valid:
        r_format = 0
    elif parsed.path == "evidence":
        if not all(quote_in_passages(q, sample.retrieved_passages) for q in parsed.quoted_evidence):
            r_format = 0

    # R_path gate: correct choice given is_answerable.
    correct_path = (
        (parsed.path == "evidence" and sample.is_answerable)
        or (parsed.path == "llm" and not sample.is_answerable)
    )
    r_path = 1 if correct_path else 0

    # R_quality in [0,1].
    r_quality = 0.0
    if r_format == 1 and r_path == 1:
        if parsed.path == "evidence":
            rl_ev = rouge_l(" ".join(parsed.quoted_evidence), sample.gold_evidence)
            rl_ans = rouge_l(parsed.answer, sample.gold_answer)
            r_quality = 0.5 * rl_ev + 0.5 * rl_ans
        else:  # llm path
            r_quality = rouge_l(parsed.answer, sample.gold_answer)

    reward = r_format * (cfg.w_p * r_path + cfg.w_q * r_path * r_quality)
    parts = {
        "path": parsed.path,
        "R_format": r_format,
        "R_path": r_path,
        "R_quality": r_quality,
    }
    return reward, parts
```

- [ ] **Step 8:** `pytest tests/test_r2.py -v`. Expected: all PASS.

- [ ] **Step 9 (TDD):** Failing tests for audit — `tests/test_audit.py`:

```python
from src.rewards.audit import audit_batch

def test_audit_computes_path_shares():
    responses = [
        ("<evidence>q</evidence><answer>a</answer>", "What is X?"),
        ("<evidence>q</evidence><answer>a</answer>", "What is Y?"),
        ("<llm>no</llm><answer>unknown</answer>", "What is Z?"),
        ("broken", "What is W?"),
    ]
    retrieved = [["q"], ["q"], ["x"], ["z"]]
    report = audit_batch(responses, retrieved)
    assert report["path_shares"]["evidence"] == 0.5
    assert report["path_shares"]["llm"] == 0.25
    assert report["path_shares"]["invalid"] == 0.25

def test_audit_flags_fabricated_quotes():
    responses = [
        ("<evidence>fabricated text</evidence><answer>a</answer>", "q"),
        ("<evidence>q</evidence><answer>a</answer>", "q"),
    ]
    retrieved = [["only this"], ["q"]]
    report = audit_batch(responses, retrieved)
    assert report["fabricated_quote_rate"] == 0.5

def test_audit_flags_question_copy():
    responses = [
        ("<evidence>q</evidence><answer>What is X?</answer>", "What is X?"),
        ("<evidence>q</evidence><answer>correct</answer>", "What is Y?"),
    ]
    retrieved = [["q"], ["q"]]
    report = audit_batch(responses, retrieved)
    assert report["question_copy_rate"] == 0.5
```

- [ ] **Step 10:** `pytest tests/test_audit.py -v`. Expected: FAIL.

- [ ] **Step 11:** Implement `src/rewards/audit.py`:

```python
"""Reward-hacking audit utilities. Run on completed rollouts to detect exploitation."""
from statistics import mean
from src.common.parse import parse_response

def audit_batch(responses_with_questions, retrieved_passages_per_sample) -> dict:
    """responses_with_questions: list[tuple[str, str]] of (response, question).
    retrieved_passages_per_sample: list[list[str]], parallel.
    """
    n = len(responses_with_questions)
    assert n == len(retrieved_passages_per_sample)
    assert n > 0

    paths = {"evidence": 0, "llm": 0, "invalid": 0}
    evidence_lengths = []
    fabricated = 0
    evidence_count = 0
    question_copy = 0

    for (resp, question), retrieved in zip(responses_with_questions, retrieved_passages_per_sample):
        parsed = parse_response(resp)
        paths[parsed.path] += 1

        if parsed.path == "evidence":
            evidence_count += 1
            evidence_lengths.append(sum(len(q.split()) for q in parsed.quoted_evidence))
            if not all(any(q.strip() in p for p in retrieved) for q in parsed.quoted_evidence):
                fabricated += 1

        if parsed.format_valid and question.strip() and question.strip() in parsed.answer:
            question_copy += 1

    return {
        "n": n,
        "path_shares": {k: v / n for k, v in paths.items()},
        "evidence_length_mean_tokens": mean(evidence_lengths) if evidence_lengths else 0.0,
        "fabricated_quote_rate": fabricated / evidence_count if evidence_count else 0.0,
        "question_copy_rate": question_copy / n,
    }
```

- [ ] **Step 12:** `pytest tests/test_audit.py -v`. Expected: all PASS.

- [ ] **Step 13 (checkpoint):** Suggested commit: `feat(rewards): R1, R2, and reward-hacking audit with full test coverage`

---

### Task 3.3 (Member 3): DAPO config + grid runner

**Files:**
- Create: `configs/r0_dapo.yaml`
- Create: `configs/r1_grpo.yaml`
- Create: `configs/r1_dapo.yaml`
- Create: `configs/r2_grpo.yaml`
- Create: `configs/r2_dapo.yaml`
- Create: `configs/grid.yaml`
- Create: `scripts/run_grid.sh`

- [ ] **Step 1:** Copy `configs/r0_grpo.yaml` → `configs/r0_dapo.yaml` and change:

```yaml
_meta:
  reward: r0
  optimizer: dapo    # changed
  model: qwen3-0.6b
  seed: 0

# ... same as r0_grpo.yaml except:
algorithm:
  name: dapo
  # DAPO-specific: decoupled clipping + dynamic sampling.
  # Use GRACE repo's verbatim DAPO config here — DO NOT tune.
  clip_ratio_low: 0.2
  clip_ratio_high: 0.28
  dynamic_sampling: true
```

(Exact DAPO field names: see verl's DAPO recipe — copy from `verl/examples/dapo_*.yaml`.)

- [ ] **Step 2:** Copy `configs/r0_grpo.yaml` → `configs/r1_grpo.yaml`, change `_meta.reward: r1` and:

```yaml
reward_fn:
  path: src.rewards.r1_cost_sensitive:r1_cost_sensitive
```

- [ ] **Step 3:** Same for `r1_dapo.yaml`, `r2_grpo.yaml`, `r2_dapo.yaml` — change `reward` and `reward_fn.path`.

- [ ] **Step 4:** Create `configs/grid.yaml`:

```yaml
# The experimental grid. run_grid.sh reads this.
runs:
  # Must-have (4)
  - config: configs/r0_grpo.yaml
    priority: must
  - config: configs/r0_dapo.yaml
    priority: must
  - config: configs/r1_grpo.yaml
    priority: must
  - config: configs/r2_grpo.yaml
    priority: must
  # Should-have (2)
  - config: configs/r1_dapo.yaml
    priority: should
  - config: configs/r2_dapo.yaml
    priority: should
```

- [ ] **Step 5:** Write `scripts/run_grid.sh`:

```bash
#!/usr/bin/env bash
# Usage: run_grid.sh [must|should|all]
# Launches runs sequentially (safer than parallel on 1 GPU) and evaluates each.
set -euo pipefail
PRIORITY=${1:-must}

source ~/rl-rag-venv/bin/activate
cd $(dirname $0)/..

configs=$(python -c "
import yaml, sys
g = yaml.safe_load(open('configs/grid.yaml'))
want = '${PRIORITY}'
for r in g['runs']:
    if want == 'all' or r['priority'] == want:
        print(r['config'])
")

for cfg in $configs; do
    echo "=== Running ${cfg} ==="
    bash scripts/launch_run.sh "${cfg}"

    # Find most recent results dir matching this config's _meta.
    LATEST=$(ls -td results/*/ | head -n 1)
    echo "=== Evaluating ${LATEST} ==="
    CKPT=$(ls -td ${LATEST}/checkpoints/global_step_* 2>/dev/null | head -n 1)
    if [[ -z "${CKPT}" ]]; then
        echo "No checkpoint found; skipping eval."
        continue
    fi
    python -m src.eval.evaluate \
        --checkpoint "${CKPT}" \
        --split data/cache/dense_val.jsonl \
        --out "${LATEST}/metrics.json"
done
```

- [ ] **Step 6:** Launch R0+DAPO as RQ2 half: `bash scripts/launch_run.sh configs/r0_dapo.yaml`. Monitor. When complete, evaluate.

- [ ] **Step 7:** Verify R0+DAPO metrics.json exists and Balanced Accuracy is in a sensible range (not necessarily matching R0+GRPO — DAPO may differ, that's the point).

- [ ] **Step 8 (checkpoint):** Suggested commit: `feat(training,experiments): DAPO config + grid runner + R0+DAPO run complete`

---

### Week 3 Gate verification

- [ ] `pytest tests/test_r1.py tests/test_r2.py tests/test_audit.py tests/test_judge.py -v` all pass
- [ ] `python scripts/validate_lj.py` prints agreement ≥ 85%
- [ ] `results/<r0_dapo_run>/metrics.json` exists
- [ ] `scripts/run_grid.sh must` executes without immediate errors (dry-run; don't launch the full grid yet)

---

## Week 4 (Nov 2–8) — Full grid execution + first analysis

**Week goal:** all 4 must-have runs have metrics; comparison table shows preliminary RQ1/RQ2 signal.

### Task 4.1 (Member 3): Launch and babysit the must-have grid

- [ ] **Step 1:** Launch the must-have grid: `bash scripts/run_grid.sh must`. This will serialize 4 runs; expect ~1-2 days wall time depending on GPU and per-run step count.
- [ ] **Step 2:** Monitor each run — tail `train.log`, watch W&B/tensorboard. If any run crashes mid-training, note the failure in `docs/log/<date>-grid-<run>.md`, fix, re-launch.
- [ ] **Step 3:** When all 4 runs complete, verify `results/*/metrics.json` exist for R0+GRPO, R0+DAPO, R1+GRPO, R2+GRPO.

### Task 4.2 (Member 1): Result aggregation + comparison table

**Files:**
- Create: `src/eval/aggregate.py`
- Create: `tests/test_aggregate.py`
- Create: `scripts/make_comparison_table.py`

- [ ] **Step 1 (TDD):** Failing test — `tests/test_aggregate.py`:

```python
import json
from pathlib import Path
from src.eval.aggregate import collect_metrics

def test_collect_metrics_reads_all_runs(tmp_path):
    for name, metrics in [
        ("r0_grpo_x_s0_20261102-120000", {"balanced_accuracy": 0.75, "em_answerable": 0.6, "abstention_unanswerable": 0.9}),
        ("r1_grpo_x_s0_20261103-120000", {"balanced_accuracy": 0.78, "em_answerable": 0.65, "abstention_unanswerable": 0.91}),
    ]:
        d = tmp_path / name
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps(metrics))
    df = collect_metrics(tmp_path)
    assert len(df) == 2
    assert set(df["reward"]) == {"r0", "r1"}
    assert set(df["optimizer"]) == {"grpo"}
```

- [ ] **Step 2:** `pytest tests/test_aggregate.py -v`. Expected: FAIL.

- [ ] **Step 3:** Implement `src/eval/aggregate.py`:

```python
"""Collect metrics across runs into a tidy table."""
from pathlib import Path
import json

def collect_metrics(results_dir: Path) -> list[dict]:
    rows = []
    for run_dir in Path(results_dir).iterdir():
        if not run_dir.is_dir():
            continue
        mf = run_dir / "metrics.json"
        if not mf.exists():
            continue
        parts = run_dir.name.split("_")
        if len(parts) < 4:
            continue
        reward, optimizer, model, seed = parts[0], parts[1], parts[2], parts[3]
        m = json.loads(mf.read_text())
        rows.append({
            "run_id": run_dir.name,
            "reward": reward,
            "optimizer": optimizer,
            "model": model,
            "seed": seed,
            **m,
        })
    return rows
```

(We return a `list[dict]` to avoid a pandas dep for one function; tests may need to adapt if DataFrame-like access is desired.)

Update the test to use list-of-dict access instead of `df["reward"]`:

```python
# tests/test_aggregate.py — rewrite:
    rows = collect_metrics(tmp_path)
    assert len(rows) == 2
    assert {r["reward"] for r in rows} == {"r0", "r1"}
    assert {r["optimizer"] for r in rows} == {"grpo"}
```

- [ ] **Step 4:** `pytest tests/test_aggregate.py -v`. Expected: PASS.

- [ ] **Step 5:** Write `scripts/make_comparison_table.py`:

```python
"""Dump a markdown + CSV comparison table of all runs."""
from pathlib import Path
import csv
from src.eval.aggregate import collect_metrics

rows = collect_metrics(Path("results"))
rows.sort(key=lambda r: (r["reward"], r["optimizer"]))

if not rows:
    print("No runs with metrics.json found in results/.")
    raise SystemExit(1)

cols = ["reward", "optimizer", "balanced_accuracy", "em_answerable",
        "abstention_unanswerable", "bert_f1_answerable", "lj_answerable",
        "path_shares"]

# CSV
out_csv = Path("docs/results-table.csv")
with out_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        row = {c: r.get(c, "") for c in cols}
        if isinstance(row.get("path_shares"), dict):
            row["path_shares"] = ";".join(f"{k}={v:.2f}" for k, v in row["path_shares"].items())
        w.writerow(row)

# Markdown
out_md = Path("docs/results-table.md")
lines = ["# Results comparison\n\n"]
lines.append("| " + " | ".join(cols) + " |\n")
lines.append("|" + "|".join(["---"] * len(cols)) + "|\n")
for r in rows:
    cells = []
    for c in cols:
        v = r.get(c, "")
        if isinstance(v, dict):
            v = "; ".join(f"{k}={vv:.2f}" for k, vv in v.items())
        elif isinstance(v, float):
            v = f"{v:.3f}"
        cells.append(str(v))
    lines.append("| " + " | ".join(cells) + " |\n")
out_md.write_text("".join(lines), encoding="utf-8")

print(f"Wrote {out_csv} and {out_md}")
```

- [ ] **Step 6:** Run `python scripts/make_comparison_table.py`. Verify `docs/results-table.md` renders.

- [ ] **Step 7 (checkpoint):** Suggested commit: `feat(eval): result aggregation + comparison table script`

### Task 4.3 (Member 2): Reward-hacking audit on completed runs

**Files:**
- Create: `scripts/run_audits.py`

- [ ] **Step 1:** Write `scripts/run_audits.py`:

```python
"""Run reward-hacking audit on all completed runs' predictions.jsonl."""
import json
from pathlib import Path
from src.rewards.audit import audit_batch

for run in Path("results").iterdir():
    pred = run / "predictions.jsonl"
    if not pred.exists():
        continue
    rows = [json.loads(l) for l in pred.open(encoding="utf-8")]
    # Also need retrieved_passages — load from the split used.
    # Convention: config.yaml records data.val_files; look it up.
    import yaml
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    val_file = Path(cfg["data"]["val_files"])
    split_rows = {json.loads(l)["sample_id"]: json.loads(l) for l in val_file.open(encoding="utf-8")}

    pairs = []
    retrieved = []
    for r in rows:
        s = split_rows[r["sample_id"]]
        # We don't have the raw response in predictions.jsonl (we only stored parsed answer).
        # TODO W5: extend evaluate.py to also save raw response. For now, reconstruct from parsed.
        # Minimal: audit over path-only data.
        resp_stub = ""
        if r["path"] == "evidence":
            resp_stub = f"<evidence>{r.get('quoted_evidence','')}</evidence><answer>{r['answer']}</answer>"
        elif r["path"] == "llm":
            resp_stub = f"<llm>disclaimer</llm><answer>{r['answer']}</answer>"
        pairs.append((resp_stub, s["question"]))
        retrieved.append(s["retrieved_passages"])

    report = audit_batch(pairs, retrieved)
    (run / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{run.name}: path_shares={report['path_shares']}, "
          f"fab_quote={report['fabricated_quote_rate']:.2f}, "
          f"q_copy={report['question_copy_rate']:.2f}")
```

- [ ] **Step 2:** Important: update `src/eval/evaluate.py` to also save the raw response in `predictions.jsonl`:

```python
# Inside evaluate(), where rows are built:
rows.append({
    "sample_id": s.sample_id,
    "is_answerable": s.is_answerable,
    "path": parsed.path,
    "quoted_evidence": parsed.quoted_evidence,
    "answer": parsed.answer,
    "raw_response": resp,       # ADD THIS
    "em": em,
})
```

Then `scripts/run_audits.py` can use `r["raw_response"]` directly instead of the stub reconstruction. Clean up the TODO in that script.

- [ ] **Step 3:** Re-run eval on existing checkpoints (just the 4 must-have) so `predictions.jsonl` includes `raw_response`. Then run `python scripts/run_audits.py`.

- [ ] **Step 4:** Open each `results/<run>/audit.json`. Flag any pattern:
  - path_shares with evidence > 0.95 or llm > 0.95 (policy collapse)
  - fabricated_quote_rate > 0.1 (fabrication exploitation)
  - question_copy_rate > 0.1 (reward hacking via copy)
  - evidence_length_mean_tokens > 50 (over-quoting to boost RLev)

Document findings in `docs/audit-week4.md`. **Iterate on R1/R2 only if a reward shows clear exploitation** — don't tune for marginal effects.

- [ ] **Step 5:** Draft the "Reward Design" section of the report in `docs/report/reward-design.md`.

- [ ] **Step 6 (checkpoint):** Suggested commit: `feat(audit): reward-hacking audit on week 4 grid results + report section`

### Task 4.4 (Member 1 stretch): Begin QASPER data prep

- [ ] **Step 1:** Write `src/data/qasper.py` by analogy to `src/data/hotpot.py`, converting QASPER raw rows into `Sample`s. QASPER is scientific-paper QA — different format: `allenai/qasper` on HF.
- [ ] **Step 2:** Extend `scripts/build_retrieval.py` to optionally build QASPER splits.
- [ ] **Step 3:** Do NOT run full retrieval on QASPER yet — Week 5 territory.

---

### Week 4 Gate verification

- [ ] 4 must-have runs each have `metrics.json`
- [ ] `docs/results-table.md` contains a row for each
- [ ] 4 corresponding `audit.json` files exist
- [ ] `docs/audit-week4.md` summarizes findings
- [ ] Preliminary RQ1 answer visible: compare `abstention_unanswerable` across R0/R1/R2 (same optimizer)
- [ ] Preliminary RQ2 answer visible: compare R0+GRPO vs R0+DAPO on `balanced_accuracy` and (if logged) sample efficiency

---

## Week 5 (Nov 9–15) — Should-runs + hero run + report draft (= target completion)

**Week goal:** Full 6-run grid in one table; Qwen3-1.7B hero run on winning combo; report skeleton with draft sections.

### Task 5.1 (Member 3): Launch R1+DAPO, R2+DAPO, and the hero run

- [ ] **Step 1:** Launch should-have grid: `bash scripts/run_grid.sh should`. Monitor.
- [ ] **Step 2:** When all 6 base-grid runs complete, inspect `docs/results-table.md`. Pick the winner by (Balanced Accuracy desc, then `abstention_unanswerable` desc).
- [ ] **Step 3:** Create `configs/hero_1.7b.yaml` by copying the winner's config and changing `model.path` to `Qwen/Qwen3-1.7B` + `total_training_steps` up by ~2× (budget-dependent).

```yaml
_meta:
  reward: <winner>
  optimizer: <winner>
  model: qwen3-1.7b
  seed: 0

# ... same as winner config except:
actor_rollout_ref:
  model:
    path: Qwen/Qwen3-1.7B
    enable_lora: true
    lora_rank: 16

trainer:
  total_training_steps: 800
```

- [ ] **Step 4:** Launch hero run: `bash scripts/launch_run.sh configs/hero_1.7b.yaml`. Evaluate when done. Expect wall time ~2× the 0.6B runs.
- [ ] **Step 5:** Add the hero run to the comparison table.
- [ ] **Step 6:** Draft "Training & Optimizer Comparison" section at `docs/report/training.md`.

### Task 5.2 (Member 1): QASPER cross-domain stretch + final tables

- [ ] **Step 1:** If time permits: `python scripts/build_retrieval.py --dataset qasper`. Then launch the winner's reward+optimizer on QASPER with a new config `configs/<winner>_qasper.yaml`.
- [ ] **Step 2:** Add QASPER row to the comparison table.
- [ ] **Step 3:** Generate per-dataset breakdown figures (bar charts: answerable EM, unanswerable abstention, balanced accuracy per reward). Use matplotlib; save to `docs/figures/`.
- [ ] **Step 4:** Write the "Results" section at `docs/report/results.md` referencing the tables and figures.

### Task 5.3 (Member 2): Finalize reward analysis + failure cases

- [ ] **Step 1:** Re-run audits on all 6 runs. Update `docs/audit-week4.md` → `docs/audit-final.md`.
- [ ] **Step 2:** Pick 3-5 illustrative failure cases per reward (hallucination cases, fabrication cases, over-refusal cases). Save to `docs/report/failure-cases.md`.
- [ ] **Step 3:** Finalize "Reward Design" + "Reward-Hacking Audit" sections of the report.

### Task 5.4 (whole team): Report assembly

- [ ] **Step 1:** Create `docs/report/REPORT.md` combining:
  - Abstract (Member 2 drafts)
  - Background (Member 2 drafts from spec §1)
  - Method:
    - Reward design (Member 2)
    - Training + optimizer (Member 3)
    - Data + retrieval (Member 1)
  - Results (Member 1 + Member 3)
  - Discussion (Member 2 leads, team edits)
  - Conclusions + future work
- [ ] **Step 2:** Each member reviews the others' sections.

---

### Week 5 Gate verification (= target completion)

- [ ] All 6 base-grid `metrics.json` files exist
- [ ] Qwen3-1.7B hero run has `metrics.json` (or explicit decision in `docs/log/` to skip it)
- [ ] `docs/results-table.md` includes all runs
- [ ] `docs/report/REPORT.md` has drafts of all sections
- [ ] `docs/audit-final.md` written

---

## Week 6 (Nov 16–22) — Buffer / stretch / ablations

If Week 5 gate passed cleanly, pursue ablations. If gate slipped, catch up.

### Task 6.1 (optional, Member 2): Hyperparameter ablation on winning reward

- [ ] **Step 1:** Pick the winning reward. Vary its one or two most-impactful hyperparameters (R1: `lam`, `gamma`; R2: `w_p`, `w_q`). Create 3-4 ablation configs.
- [ ] **Step 2:** Launch sequentially via `scripts/launch_run.sh` for each config.
- [ ] **Step 3:** Add ablation rows to the comparison table.
- [ ] **Step 4:** Write "Ablation" subsection in `docs/report/results.md`.

### Task 6.2 (optional, Member 1): Full QASPER run if not done in Week 5

- [ ] **Step 1:** Launch QASPER run.
- [ ] **Step 2:** Add to table.
- [ ] **Step 3:** Write "Cross-domain generalization" subsection in `docs/report/discussion.md`.

### Task 6.3 (optional, Member 2 + Member 3): Error analysis

- [ ] **Step 1:** Sample 20 failure cases per run from `predictions.jsonl`. Classify each: hallucination / over-refusal / wrong-quote / correct-reasoning-wrong-answer.
- [ ] **Step 2:** Aggregate into a table in `docs/report/error-analysis.md`.

---

## Week 7 (Nov 23–29) — Final polish + submit

- [ ] **Step 1:** Every member proofreads the full `docs/report/REPORT.md`. Fix typos, tighten claims, verify every number cites a table/figure.
- [ ] **Step 2:** Generate the final PDF (e.g., `pandoc docs/report/REPORT.md -o REPORT.pdf` or copy into LaTeX template).
- [ ] **Step 3:** Verify every citation in the References section resolves.
- [ ] **Step 4:** Dry-run presentation if the course requires one.
- [ ] **Step 5:** Submit before end of November.

---

## Appendix A — Per-member responsibility recap

| Week | Member 1 (Data/Eval) | Member 2 (Rewards) | Member 3 (Training) |
|---|---|---|---|
| 0 | repo scaffold, env setup | — | — |
| 1 | `src/common/types.py`, `src/common/parse.py`, `src/data/hotpot.py` tiny loader | prompts, R0 (full), R0 unit tests, ROUGE-L helpers | launcher promotion, run_naming, results convention |
| 2 | retrievers, balanced split, retrieval cache, full download | R0 edge-case tests, R1/R2 design doc | full GRPO config, evaluate.py (EM/BERTScore skeleton), R0+GRPO run |
| 3 | Groq LJ + cache + 50-example validation, BERTScore wiring | R1, R2, audit utilities | DAPO configs, grid runner, R0+DAPO run |
| 4 | result aggregation, comparison table, QASPER prep (stretch) | audits on completed runs, report section | launch + babysit must-grid |
| 5 | QASPER run (stretch), final tables + figures | audit finalization, failure cases, report sections | should-grid runs + hero Qwen3-1.7B run |
| 6 | stretch QASPER if needed | ablation on winning reward (optional) | error analysis support |
| 7 | report polish | report polish | report polish |

## Appendix B — File inventory

Each file created by this plan, grouped by lane:

**Shared (`src/common/`, created in Week 1):**
- `src/common/types.py` — `Sample`, `ParsedResponse`
- `src/common/parse.py` — `parse_response()`

**Member 1 (data + eval):**
- `src/data/hotpot.py`, `src/data/qasper.py`, `src/data/retrievers.py`, `src/data/balanced_split.py`, `src/data/cache.py`
- `src/eval/metrics.py`, `src/eval/judge.py`, `src/eval/evaluate.py`, `src/eval/aggregate.py`
- `scripts/download_hotpot_tiny.py`, `scripts/download_hotpot_full.py`, `scripts/build_retrieval.py`, `scripts/make_lj_validation.py`, `scripts/validate_lj.py`, `scripts/make_comparison_table.py`
- `data/lj_validation/labels.jsonl`
- `tests/test_hotpot.py`, `tests/test_retrievers.py`, `tests/test_balanced_split.py`, `tests/test_eval_metrics.py`, `tests/test_judge.py`, `tests/test_aggregate.py`

**Member 2 (rewards):**
- `src/prompts/templates.py`
- `src/rewards/metrics.py`, `src/rewards/r0_grace.py`, `src/rewards/r1_cost_sensitive.py`, `src/rewards/r2_hard_gated.py`, `src/rewards/audit.py`
- `scripts/run_audits.py`
- `tests/test_r0.py`, `tests/test_r1.py`, `tests/test_r2.py`, `tests/test_metrics.py`, `tests/test_audit.py`

**Member 3 (training):**
- `src/training/run_naming.py`
- `configs/skeleton.yaml`, `configs/r0_grpo.yaml`, `configs/r0_dapo.yaml`, `configs/r1_grpo.yaml`, `configs/r1_dapo.yaml`, `configs/r2_grpo.yaml`, `configs/r2_dapo.yaml`, `configs/grid.yaml`, `configs/hero_1.7b.yaml`
- `scripts/setup_env.sh`, `scripts/launch_skeleton.sh`, `scripts/launch_run.sh`, `scripts/run_grid.sh`
- `tests/test_run_naming.py`

**Docs:**
- `docs/team.md`, `docs/data.md`, `docs/rewards.md`, `docs/results-convention.md`, `docs/results-table.md`, `docs/results-table.csv`, `docs/audit-week4.md`, `docs/audit-final.md`, `docs/log/*.md`, `docs/report/*.md`, `docs/figures/*`

---

## Appendix C — Gate cheat-sheet

| Week | Gate (all must pass to proceed) |
|---|---|
| 1 | pytest green on each laptop; 1-step run completes on server; shared contract used in all 3 lanes; sync mechanism documented |
| 2 | retrieval cache built with documented ratios; R0 unit tests cover all branches; full R0+GRPO run completes; **Balanced Accuracy ∈ [0.71, 0.91]** (else whole team halts) |
| 3 | R1/R2 unit tests pass; LJ agreement ≥ 85%; R0+DAPO metrics.json exists; grid runner dry-runs cleanly |
| 4 | 4 must-grid metrics.json exist; comparison table written; 4 audit.json exist; preliminary RQ1/RQ2 answers visible |
| 5 | 6 base-grid metrics.json exist; hero run done (or decision not to); report draft covers all sections; audit-final.md written |

---

End of plan.
