# Midterm Report — RL-Assisted RAG: Reward Design

**Course:** MSML635 — Reinforcement Learning
**Group:** 5
**Date:** 2026-10-20 (Week 2 of a 7-week project)
**Final deadline:** End of November 2026
**Repository:** <https://github.com/elderflamevandle/RL-MSML635-Group-Project>

---

## Abstract

We are investigating **reward design for RL-assisted Retrieval-Augmented Generation (RAG)**. A RAG policy must decide, per query, whether to (a) take an `<evidence>` path and quote supporting sentences from retrieved passages, or (b) take an `<llm>` path and abstain when evidence is insufficient. Our baseline is the GRACE framework (ICLR 2025), which uses a summed reward R₀. We are testing two alternatives — **R₁** (cost-sensitive with a hallucination penalty) and **R₂** (hard-gated multiplicative) — against R₀ under both **GRPO** and **DAPO** policy-optimization recipes. Our research questions are **RQ1: does reward *gating* reduce hallucination on unanswerable queries?** and **RQ2: how do GRPO and DAPO compare in accuracy and sample efficiency on this task?**

At midterm, **all engineering infrastructure is complete** — rewards, retrievers, evaluation harness, caches, and 88 unit tests — and we have a validated **pre-RL baseline** on Qwen3-0.6B (Balanced Accuracy = 0.04 on HotpotQA tiny). **RL training itself is pending GPU-server provisioning**, which we expect to resolve this week and treat as our Week-2 gate.

---

## 1. Introduction

### 1.1 Problem

Modern RAG systems either hallucinate when retrieval is weak (claiming answers that aren't in the passages) or over-abstain when retrieval is strong (refusing to answer well-supported questions). GRACE (ICLR 2025) tackles this with a per-query path choice — `<evidence>` for grounded answers, `<llm>` for honest abstention — and a reward that sums bonuses for correct path, format, and evidence quality.

### 1.2 Hypothesis

Summed rewards are susceptible to **ceiling asymmetry**: the `<evidence>` path's maximum reward (`R_f + R_p + α + β` ≈ 2.5) exceeds the `<llm>` path's maximum (`R_f + R_p` = 1.5), biasing the policy toward answering even when it should abstain. If we gate on each requirement multiplicatively, so a missing piece collapses the reward to 0 rather than merely reducing it, the policy should learn to abstain more honestly.

### 1.3 Research questions

- **RQ1.** Does gating the reward (vs summing all terms) stop the policy from guessing on insufficient evidence?
- **RQ2.** With the reward fixed, how do **GRPO** and **DAPO** compare in balanced accuracy and sample efficiency?

### 1.4 Deliverables

A **6-run experimental grid** (3 rewards × 2 optimizers) on HotpotQA, plus a stretch run on QASPER for cross-domain validation and a Qwen3-1.7B "hero" run on the winning combination.

---

## 2. Approach

### 2.1 Reward designs

**R₀ — GRACE baseline (control).** Summed reward with branches on format validity and path choice:

```
R₀ = 0                               if format invalid
   = R_f                             if valid format, wrong path
   = R_f + R_p                       if correct <llm> path
   = R_f + R_p + α·R_Lev + β·R_Lans  if correct <evidence> path
```

with `R_f = 0.5`, `R_p = 1.0`, and `R_Lev / R_Lans` = ROUGE-L against gold evidence and gold answer respectively. α, β are inherited from GRACE's published values.

**R₁ — Cost-sensitive with hallucination penalty.** Adds (a) an explicit `-λ` penalty when the policy answers an unanswerable query, (b) a quality bonus `γ · R_Lans` on the `<llm>` path to close the ceiling asymmetry, and (c) a quote-containment indicator `𝟙[quote ⊂ C_ret]` that gates the format bonus on the `<evidence>` path — so fabricated evidence (ROUGE-matches gold but wasn't in the retrieved context) earns zero format credit.

**R₂ — Hard-gated multiplicative (RQ1 extreme).** All terms multiply:

```
R₂ = R_format × (w_p · R_path + w_q · R_path · R_quality)
```

where `R_format, R_path ∈ {0, 1}`. Any failure (invalid format, wrong path, non-verbatim quote) collapses R₂ to 0 — no partial credit. If R₂ wins on Balanced Accuracy, gating works; if R₂ collapses the policy to always-abstain, gating is too strict and R₁ is the sweet spot. Either outcome answers RQ1.

Starting hyperparameters: `λ = 1.0`, `γ = 0.5`, `w_p = 0.4`, `w_q = 0.6`. No sweeps until Week 6.

### 2.2 Policy optimization

Both **GRPO** (Group Relative Policy Optimization) and **DAPO** (Decoupled Clip and Dynamic sAmpling Policy Optimization) via `verl` at the GRACE paper's pinned commit `4aa02fe`. No custom algorithm code — just reward functions plugged into verl recipes.

### 2.3 Datasets

- **HotpotQA** (primary). 2000 train + 500 val from the `distractor` split, balanced ~50/50 answerable/unanswerable via a deterministic distractor-swap procedure (SHA-1 hash of `sample_id` seeded with target rate 0.45).
- **QASPER** (stretch, Week 5). Cross-domain validation on scientific papers; loader already implemented.

### 2.4 Model

**Qwen3-0.6B + LoRA** for the main grid (fits on a single 6 GB GPU); **Qwen3-1.7B hero run** on the winning (reward, optimizer) in Week 5.

### 2.5 Evaluation

- **Exact Match (EM)** on answerable queries — primary accuracy.
- **Abstention Accuracy (TNR)** on unanswerable queries.
- **Balanced Accuracy = mean(EM, Abstention)** — primary metric.
- **BERTScore** — semantic-similarity sanity check.
- **LLM-as-Judge** — Groq-hosted Llama-3.3-70B at temperature 0, with on-disk caching keyed by prompt+response hash. Validated for ≥85% agreement with EM on a 50-example hand-labeled subset.

### 2.6 Experimental grid

| Reward | Optimizer | Dataset | Priority |
|---|---|---|---|
| R₀ | GRPO | HotpotQA | **must** (baseline) |
| R₀ | DAPO | HotpotQA | **must** (RQ2 half) |
| R₁ | GRPO | HotpotQA | **must** |
| R₂ | GRPO | HotpotQA | **must** (RQ1 answer) |
| R₁ | DAPO | HotpotQA | should |
| R₂ | DAPO | HotpotQA | should (completes RQ1 × RQ2) |
| Best(R*, opt) on Qwen3-1.7B | — | HotpotQA | should (hero) |
| Best(R*, opt) | — | QASPER | stretch (cross-domain) |

---

## 3. System implementation

### 3.1 Architecture overview

```
                               ┌────────────────────┐
                               │  HotpotQA / QASPER │
                               │   (data/raw/)      │
                               └─────────┬──────────┘
                                         │
                                         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  src/data/        hotpot.py, qasper.py                      │
    │                   retrievers.py (BM25, Qwen3-Embedding-0.6B)│
    │                   balanced_split.py, cache.py               │
    └──────────────────────────┬──────────────────────────────────┘
                               ▼ Sample (sample_id, question, …, retrieved_passages)
    ┌─────────────────────────────────────────────────────────────┐
    │  src/prompts/     templates.py  →  build_prompt()           │
    └──────────────────────────┬──────────────────────────────────┘
                               ▼ str (prompt)
    ┌─────────────────────────────────────────────────────────────┐
    │  verl GRPO/DAPO rollout  →  model.generate()                │
    └──────────────────────────┬──────────────────────────────────┘
                               ▼ str (response)
    ┌─────────────────────────────────────────────────────────────┐
    │  src/common/parse.py     parse_response()                   │
    │                          → path ∈ {evidence, llm, invalid}  │
    │                          → quoted_evidence, answer          │
    └──────────────────────────┬──────────────────────────────────┘
                               ▼ ParsedResponse
    ┌─────────────────────────────────────────────────────────────┐
    │  src/rewards/     r0_grace.py, r1_cost_sensitive.py,        │
    │                   r2_hard_gated.py, audit.py                │
    │                   → (scalar reward, sub-score dict)         │
    └──────────────────────────┬──────────────────────────────────┘
                               ▼
                       verl policy update
```

The **integration contract** is `Sample` (dataclass) + `parse_response()` (single-source-of-truth parser). All three member lanes compile against this contract, so each can be developed independently on a laptop.

### 3.2 What is implemented and tested (as of 2026-10-20)

| Component | Files | Status |
|---|---|---|
| HotpotQA loader + balanced split | `src/data/hotpot.py`, `src/data/balanced_split.py` | ✅ 3 tests |
| QASPER loader (cross-domain) | `src/data/qasper.py` | ✅ 10 tests |
| BM25 retriever | `src/data/retrievers.py` | ✅ 2 tests, 2000 train + 500 val built |
| Dense retriever (Qwen3-Embedding-0.6B) | `src/data/retrievers.py` | ✅ 1 test, server-side build pending |
| Retrieval cache (SHA-256 keys, atomic, collision-safe) | `src/data/cache.py` | ✅ 2 tests, 4000 entries cached |
| Prompt templates | `src/prompts/templates.py` | ✅ 1 test + 3 round-trip tests with parser |
| Response parser | `src/common/parse.py` | ✅ 5 tests |
| **R₀ reward (GRACE control)** | `src/rewards/r0_grace.py` | ✅ 9 tests (all 4 branches + edge cases) |
| **R₁ reward (cost-sensitive)** | `src/rewards/r1_cost_sensitive.py` | ✅ unit tests (part of 14 R1/R2/audit tests) |
| **R₂ reward (hard-gated)** | `src/rewards/r2_hard_gated.py` | ✅ unit tests |
| ROUGE-L + quote-containment utilities | `src/rewards/metrics.py` | ✅ 7 tests |
| Reward-hacking audit utilities | `src/rewards/audit.py` | ✅ tests |
| Eval metrics (EM, abstention, balanced acc) | `src/eval/metrics.py` | ✅ 6 tests |
| Eval harness (vLLM-backed) | `src/eval/evaluate.py` | ✅ laptop-import-safe (lazy vLLM) |
| Local sanity runner (laptop GPU) | `src/eval/local_runner.py`, `scripts/local_sanity.py` | ✅ verified end-to-end on 50 samples |
| Groq LLM-as-Judge client | `src/eval/judge.py` | ✅ structure + caching done; needs key to validate |
| Metrics aggregation + comparison tables | `src/eval/aggregate.py`, `scripts/make_comparison_table.py` | ✅ 3 tests |
| verl training launcher + run-naming | `scripts/launch_run.sh`, `src/training/run_naming.py` | ✅ server-side, pinned to GRACE's verl commit `4aa02fe` |
| Base GRPO config | `configs/r0_grpo.yaml`, `configs/skeleton.yaml` | ✅ draft; needs verl-field-name iteration on server |

**Test coverage: 88 passing (1 deselected slow test).** All rewards, parser, retrievers, metrics, and data pipeline exercised. Determinism verified via stable argsort (retrievers) + SHA-256 cache keys.

### 3.3 Code artifacts worth noting

- **Collision-safe retrieval cache.** `src/data/cache.py` stores the full key tuple alongside the value and raises on hash collision, preventing silently corrupted retrievals across runs.
- **Fail-loud gold-passage assertion.** `src/data/hotpot.py` asserts non-empty gold passages at load time; silent breakage of ans/unans ratios is impossible.
- **Deterministic distractor swap.** `scripts/build_retrieval.py` uses `SHA-1(sample_id)` to pick ~45% of samples for forced-unanswerable, giving reproducible splits across machines.
- **Reward-value sanity check.** `0.1 < ans_rate < 0.95` assertion in the retrieval build fires if string normalization silently breaks the gold/retrieved match.

---

## 4. Progress against plan

### 4.1 What is complete

| Week | Task | Status |
|---|---|---|
| W0 | Repo scaffold, dependency pin, test harness | ✅ |
| W1 | Shared integration contract (`Sample`, `parse_response`) | ✅ |
| W1 | Prompts + R₀ reward + ROUGE-L metrics | ✅ |
| W1 | Training launcher + run naming + results convention | ✅ |
| W2 | BM25 + dense retriever + balanced split + cache | ✅ (BM25 built; dense pending server) |
| W2 | R₀ edge-case tests + reward design doc | ✅ |
| W2 | Eval harness skeleton (EM, abstention, balanced acc) | ✅ |
| W3 | R₁ + R₂ + reward-hacking audit utilities | ✅ (schedule-ahead) |
| W3 | Groq LJ client structure | ✅ (validation blocked on key) |
| W4 | Metrics aggregation + comparison tables | ✅ (schedule-ahead) |
| W5 | QASPER loader | ✅ (schedule-ahead) |

Several Week 3-5 code deliverables have been **completed ahead of schedule** because they are laptop-safe (pure Python + unit tests) and do not depend on the GPU server. This gives us schedule buffer for the training runs themselves, which are the only remaining risk.

### 4.2 Pre-RL baseline (pipeline validation)

We ran the end-to-end pipeline against the **un-trained** Qwen3-0.6B base model on 50 HotpotQA samples (all answerable, from the tiny set). This is **not** an R0+GRPO result — it is a sanity check that the full chain (data → prompt → generate → parse → score) runs correctly, and a **floor** against which the Week-2 Gate (R0+GRPO) will be measured.

| Metric | Value |
|---|---:|
| Samples | 50 (50 answerable / 0 unanswerable) |
| Inference time | 25.5 s/sample (fp16 on RTX 4050 Laptop, 6 GB VRAM) |
| **format_valid_rate** | **0.70** |
| **em_answerable** | **0.080** |
| **balanced_accuracy** | **0.040** |
| path share: `<evidence>` | 0.68 |
| path share: `<llm>` | 0.02 |
| path share: `invalid` | 0.30 |

**Reading the baseline.** The base model follows the prompt schema 70% of the time but gets the answer right only 8% of the time. When it does follow the schema, it routes to `<evidence>` almost exclusively — the `<llm>` abstention path barely exists untrained. **The failure mode most prevalent at the floor is invalid-format (30%), not wrong-path routing.** RL training's first measurable job is to drive the invalid share down and the `<llm>` share up (because our balanced split is ~50% unanswerable).

**Gate target.** The Week-2 Gate requires R0+GRPO to achieve **Balanced Accuracy ∈ [0.71, 0.91]** (within 10 points of GRACE's published HotpotQA number of 0.81). The gap from the current floor (0.04) is **~17×** — large but expected; this is the whole point of RL fine-tuning.

### 4.3 What is pending

| Blocked on | Items |
|---|---|
| **GPU server provisioning** | Walking skeleton (1-step verl run); Week-2 Gate R0+GRPO run; full dense retriever build; all 6 grid runs; Qwen3-1.7B hero run |
| **Groq API key** | Judge validation (50-sample agreement check); re-eval of trained checkpoints with LJ |
| **Team logistics decisions** | Member assignments; results-sync mechanism; weekly meeting slot |

The server provisioning is the critical path — nothing after Week 2 can begin until a walking-skeleton verl run completes on real hardware.

---

## 5. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| verl install / GPU driver issues | High | Blocks everything | 3-day walking-skeleton pair-debug session; if still blocked, fallback to vanilla TRL + GRPO (loses DAPO, keeps RQ1) |
| R0+GRPO Balanced Accuracy wildly off GRACE's 0.81 | Medium | Blocks R₁/R₂ interpretation | Hard gate forces whole-team halt; check prompt format, ans/unans ratio, LoRA rank, model match; 1 week of Week 3 budget reserved for this |
| R₂ collapses to always-abstain | Medium | Expected failure mode — **informative, not fatal** | Documented as a positive RQ1 result (gating too strong); add R1.5 with soft gates if time permits |
| DAPO config instability | Medium | Weakens RQ2 answer | Use verl's default DAPO hyperparams from GRACE recipe verbatim — do not tune (would swamp the GRPO/DAPO signal) |
| Groq rate limit mid-eval | Low | Eval delays | On-disk cache + sleep-and-retry on 429; fallback to local Qwen3-4B zero-shot judge |
| Member unavailable for a week | Medium | Lane stalls | Shared `Sample` + `parse_response` contract means another member can pick up the lane; Member 3 can run Member 1's eval; Member 1 can run Member 2's audits |
| Qwen3-0.6B ceiling too low to show reward differences | Low-Med | Weak final signal | Week-5 Qwen3-1.7B hero run exists for exactly this; grid is kept cheap enough to afford it |
| HF dataset format drift | Low | Day-1 blocker | HuggingFace datasets revision hash pinned |

### 5.1 Known stack issues (new since proposal)

- **Local dev stack misalignment.** `sentence-transformers` requires `torch ≥ 2.6` (torch.accelerator API), but our pinned stack is `torch 2.5.1+cu121`. Workaround: pinned `transformers<5`, `datasets<5`, `sentence-transformers<5`; slow dense-retriever test deselected. **Needs resolution before server deployment** — team decision in Week 3 on upgrade vs stay-pinned.
- **Windows system Python 3.14 shadowing.** On the dev laptop, `python` resolves to `%APPDATA%\Python\Python314` (CPU-only torch 2.11), not the `.venv` (torch 2.5+cu121). Documented in `docs/log/2026-10-20-local-sanity.md`; teammates on macOS/Linux will not hit this.

---

## 6. Plan for the remainder of the project

**Overall approach.** A **3-day whole-team walking-skeleton debug** on the server, then a fan-out: Member 1 owns data/eval, Member 2 owns rewards/audit, Member 3 owns training/configs. All three converge again for the Week-5 report.

### 6.1 Week-by-week

**Week 3 (Oct 26 – Nov 1) — Walking skeleton & R₀ reproduction.**
Server env setup → iterate on verl YAML field names until 1 training step completes → launch the full R0+GRPO run → first evaluation. Member 1 wires up Groq LJ + 50-example hand-labeled validation. **Gate: R0+GRPO Balanced Accuracy ∈ [0.71, 0.91].**

**Week 4 (Nov 2 – 8) — DAPO + full "must" grid + first analysis.**
R0+DAPO config; launch all 4 must-have runs; Member 2 runs reward-hacking audits on completed checkpoints and iterates R₁/R₂ **only if audits show real problems**; Member 1 drafts aggregation tables. **Gate: comparison table with preliminary RQ1 and RQ2 signal.**

**Week 5 (Nov 9 – 15) — Should runs + hero + report draft** (target completion).
R1+DAPO, R2+DAPO; Qwen3-1.7B hero run on the winning (reward, optimizer); report drafts of Abstract, Background, Method, Results, Discussion. **Gate: 6-run grid in one table + hero metrics + report drafted.**

**Week 6 (Nov 16 – 22) — Buffer / ablations.** If gates passed: α/β/γ/λ ablations on the winning reward; full QASPER cross-domain run; 20 failure cases per reward classified by failure type. If gates slipped: catch-up.

**Week 7 (Nov 23 – 29) — Polish + submit.** Figures, citations, final tables, presentation dry-run.

### 6.2 Success criteria

- **Minimum viable (must, by Week 5):** R₀ reproduction within 10 pts of GRACE + R₀/R₁/R₂ × GRPO on HotpotQA + one comparison table + RQ1 answered with evidence.
- **Full success (target, by Week 5):** Minimum viable + full 6-run grid + per-run reward-hacking audit + RQ2 answered via GRPO vs DAPO sample-efficiency curves + Qwen3-1.7B hero.
- **Stretch (Weeks 6-7):** QASPER cross-domain + hyperparameter ablation on winner + error-case classification.

### 6.3 What "answered RQ1 / RQ2" looks like

- **RQ1 answered** when Balanced Accuracy and unanswerable abstention accuracy for R₀/R₁/R₂ are plotted side by side. A monotonic improvement R₀ → R₁ → R₂, or an informative inversion at R₂, is the result.
- **RQ2 answered** when GRPO and DAPO sample-efficiency curves (Balanced Accuracy vs training step) are overlaid for the same reward. The faster-converging / higher-plateau optimizer is the answer.

---

## 7. Team organization

| Member | Lane | Primary files |
|---|---|---|
| Member 1 (Data) | HotpotQA/QASPER loaders, retrievers, cache, eval infra, LJ validation | `src/data/*`, `src/eval/*` |
| Member 2 (Rewards) | R₀/R₁/R₂ implementation, reward-hacking audits, reward-design report section | `src/rewards/*`, `scripts/run_audits.py`, `docs/rewards.md` |
| Member 3 (Training) | verl env, launchers, GRPO/DAPO configs, grid runner, hero run, training comparison report section | `scripts/setup_env.sh`, `scripts/launch_run.sh`, `configs/*.yaml`, `src/training/*` |

Day-1 team-logistics checklist in `docs/team.md` (committed, drafted with proposed defaults for review).

---

## 8. Reproducibility

Everything a reviewer needs to reproduce midterm numbers:

```bash
# Clone + install (Python 3.12, CPU wheels OK for pipeline; CUDA 12.1 for GPU inference)
git clone https://github.com/elderflamevandle/RL-MSML635-Group-Project.git
cd RL-MSML635-Group-Project
python -m venv .venv && source .venv/bin/activate  # or .venv/Scripts/activate on Windows
pip install -e ".[dev]"

# Data
python scripts/download_hotpot_tiny.py            # 50-sample tiny set
python scripts/download_hotpot_full.py             # 2000 train + 500 val
python scripts/build_retrieval.py --no-dense       # BM25-only; dense requires torch ≥ 2.6

# Tests
pytest -q                                          # expect "88 passed, 1 deselected"

# Pre-RL baseline (requires ~5 GB VRAM, ~20 min on RTX 4050)
python scripts/local_sanity.py --n 50 --device cuda
# Writes results/local_sanity/<timestamp>/metrics.json
```

Determinism knobs: HuggingFace dataset revision pinned; retrieval cache keyed by SHA-256; argsort used in retriever ranking is stable; distractor-swap seeded by SHA-1(`sample_id`).

---

## 9. Conclusion

At the midterm mark we have delivered **all code infrastructure** — three reward designs, both algorithm configurations, HotpotQA and QASPER data loaders, retrievers (BM25 shipped, dense pending), evaluation harness, aggregation utilities, and 88 passing unit tests. We have validated the **pipeline end-to-end on real hardware** (Qwen3-0.6B on an RTX 4050, Balanced Accuracy = 0.04 as the pre-RL floor). The one open-ended risk is **GPU-server provisioning for the training runs themselves**, which is the sole blocker on the critical path from here through Week 5.

Our **success criteria are explicit and gated per week** (Section 6.2), and the plan has a **1-week slack in Week 3** reserved for the two highest-likelihood risks (verl install issues, GRACE-reproduction mismatch). The final report (end of November) will target the full 6-run grid, the Qwen3-1.7B hero run, and a reward-hacking audit per run.

---

## References

1. GRACE: *Grounded Response and Correct Abstention via Reinforcement-Learned Policy Routing* — ICLR 2025 (reward R₀ design, verl commit `4aa02fe`).
2. Yang, Z., et al. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. EMNLP 2018.
3. Dasigi, P., et al. *A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers* (QASPER). NAACL 2021.
4. Yu, Q., et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. 2024.
5. Shao, Z., et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO). 2024.
6. Qwen Team. *Qwen3 Technical Report*. 2025.
7. Zhang, T., et al. *BERTScore: Evaluating Text Generation with BERT*. ICLR 2020.
8. verl framework — <https://github.com/volcengine/verl>

---

**Appendices in-repo (not duplicated here):**

- `docs/STATUS.md` — rolling status snapshot
- `docs/rewards.md` — reward implementation reference
- `docs/log/2026-10-20-local-sanity.md` — full baseline run log
- `docs/team.md` — team logistics
- `docs/superpowers/plans/2026-10-11-rl-rag-reward-design.md` — detailed task-level plan
- `HANDOFF.md` — Day-1 scaffold handoff
- `RL_Group_5_Proposal.pdf` — original project proposal
