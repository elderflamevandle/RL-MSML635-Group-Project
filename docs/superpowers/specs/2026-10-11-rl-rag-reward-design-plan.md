# RL-Assisted RAG — Reward Design Project Plan

**Course:** MSML635 — Reinforcement Learning
**Group:** 5
**Date:** 2026-10-11
**Hard deadline:** End of November 2026 (approx. 7 weeks)
**Internal target:** End of Week 5 (2026-11-15)

---

## 1. Context

We are implementing the project described in `RL_Group_5_Proposal.pdf`: reward design for RL-assisted RAG. The policy must learn to either (a) take an `<evidence>` path and quote supporting sentences from retrieved passages, or (b) take an `<llm>` path and abstain when evidence is absent. The baseline is the GRACE framework (reward R0 + verl's DAPO recipe). The research questions are:

- **RQ1:** Does gating the reward (vs summing all terms) stop the policy from guessing on insufficient evidence?
- **RQ2:** With the reward fixed, how do GRPO and DAPO compare in accuracy and sample efficiency?

## 2. Goal

Beat our own GRACE R0 baseline on **Balanced Accuracy** on HotpotQA by designing and evaluating two reward variants (R1, R2) that attack the dominant failure mode (hallucination on unanswerable queries), and produce defensible answers to RQ1 and RQ2.

## 3. Key decisions (locked in during brainstorming)

| Area | Decision |
|---|---|
| Primary goal | Accuracy / Balanced Accuracy improvement via reward design |
| Compute | Dedicated GPU server for training; laptop for dev with CPU-mocked tests |
| Timeline | 5-week target, 7-week hard deadline (end of November) |
| Member labels | Generic Member 1 / Member 2 / Member 3 |
| Model | Qwen3-0.6B + LoRA (main grid); Qwen3-1.7B hero run as stretch |
| Rewards | R0 (GRACE control), R1 (cost-sensitive + hallucination penalty), R2 (hard-gated multiplicative) |
| Optimizers | GRPO + DAPO via verl recipes (no new algorithm code) |
| Datasets | HotpotQA primary (must), QASPER cross-domain (stretch) |
| Evaluation | EM + BERTScore + Groq-hosted Llama 3.3 70B as LLM-Judge (with on-disk response cache) |
| Primary metric | Balanced Accuracy = mean(answerable EM, unanswerable abstention accuracy) |
| Overall approach | **A+C hybrid**: 3-day walking skeleton with the whole team, then fan out to Approach A (reproduce R0 before innovating) |

## 4. Reward variants

### R0 — GRACE baseline (control)

As specified in the proposal:

```
R0 = 0,                                            invalid format
     Rf,                                           valid format, wrong path
     Rf + Rp,                                      correct <llm> path
     Rf + Rp + α·RLev + β·RLans,                   correct <evidence> path
```

with Rf = 0.5, Rp = 1, RLev/RLans = ROUGE-L vs gold. α and β match GRACE's published values.

**Known weaknesses R1/R2 attack:**
1. Ceiling asymmetry: `<evidence>` ceiling (Rf+Rp+α+β) > `<llm>` ceiling (Rf+Rp) → systemic bias toward answering.
2. No explicit cost for hallucination on unanswerable queries.
3. Nothing prevents fabricated evidence (ROUGE-matches gold without checking the quote came from Cret).

### R1 — Cost-sensitive reward with hallucination penalty

```
R1 = -λ,                                              hallucinated (answered on unanswerable)
      0,                                              invalid format
      Rf,                                             valid format, wrong path (non-hallucination)
      Rf + Rp + γ·RLans,                              correct <llm> path  (quality bonus — closes ceiling asymmetry)
      Rf·𝟙[quote ⊂ Cret] + Rp + α·RLev + β·RLans,   correct <evidence> path (quote-containment gate)
```

Starting values: λ = 1.0, γ = 0.5. α and β are inherited from R0 — read once from the GRACE paper / reference code in Week 2 when R0 is instantiated.

### R2 — Hard-gated multiplicative reward (RQ1 extreme test)

```
R2 = Rformat × (w_p·Rpath + w_q · Rpath · Rquality)

where:
  Rformat   ∈ {0, 1}    — valid tags AND verbatim-quoted evidence
  Rpath     ∈ {0, 1}    — correct <evidence>/<llm> choice
  Rquality  ∈ [0, 1]    — evidence path: 0.5·RLev + 0.5·RLans
                        — llm path:      RLans (disclaimer quality)
```

Starting values: w_p = 0.4, w_q = 0.6. Every upstream gate failing collapses R2 to 0 — no partial credit for "right path, wrong evidence."

**Interpretation:** if R2 wins on Balanced Accuracy (especially on the unanswerable subset), gating works. If R2 collapses the policy to always-abstain, gating is too strict and R1 is the sweet spot. Either outcome answers RQ1.

### Experimental grid

| Reward | Optimizer | Dataset | Priority |
|---|---|---|---|
| R0 | GRPO | HotpotQA | **must** (baseline) |
| R0 | DAPO | HotpotQA | **must** (RQ2 half) |
| R1 | GRPO | HotpotQA | **must** |
| R2 | GRPO | HotpotQA | **must** (RQ1 answer) |
| R1 | DAPO | HotpotQA | should |
| R2 | DAPO | HotpotQA | should (completes RQ1 × RQ2) |
| Best(R*, opt) on Qwen3-1.7B | — | HotpotQA | should (hero run) |
| Best(R*, opt) | — | QASPER | stretch (cross-domain) |

"Best" = highest HotpotQA Balanced Accuracy from the main grid, ties broken by unanswerable abstention accuracy.

## 5. System architecture

### Modules

| Module | Owner | Responsibility |
|---|---|---|
| `src/data/` | Member 1 | HotpotQA + QASPER loaders; balanced ans/unans split; BM25 + Qwen3-Embedding-0.6B retrievers; retrieval cache; emits normalized `Sample` records. |
| `src/prompts/` | Member 2 | Prompt templates with `<evidence>`/`<llm>` tag contract. Single source of truth imported by both rewards and training. |
| `src/rewards/` | Member 2 | Pure Python functions: `r0_grace()`, `r1_cost_sensitive()`, `r2_hard_gated()`. Each returns scalar + sub-score dict. No framework imports. |
| `src/eval/` | Member 1 | EM, BERTScore, abstention accuracy, Balanced Accuracy; Groq LJ client with on-disk cache; `evaluate.py` entrypoint. |
| `src/training/` | Member 3 | verl configs (pinned commit from GRACE); vLLM rollout setup; launch scripts; logging. |
| `src/experiments/` | Member 3 | Per-run YAML configs; `run_grid.sh`; result aggregation into comparison tables. |
| `src/common/` | shared | `Sample` dataclass + `parse_response()` function — the integration contract. |
| `tests/` | all | Reward unit tests (Member 2); data pipeline tests (Member 1); CPU smoke test for one training step (Member 3). |

### Integration contract (prevents late-integration pain)

```python
# src/common/types.py
@dataclass
class Sample:
    sample_id: str
    question: str
    gold_passages: list[str]
    distractor_passages: list[str]
    gold_answer: str
    gold_evidence: list[str]
    retrieved_passages: list[str]
    is_answerable: bool  # gold ⊆ retrieved

# src/common/parse.py
def parse_response(text: str) -> ParsedResponse:
    # Returns path ∈ {"evidence","llm","invalid"},
    # quoted_evidence: list[str], answer: str, format_valid: bool
```

Both reward functions and the eval harness call `parse_response`. One parser, one bug surface.

### Repository layout

```
F:\Reinforcement_Learning_Project\
├── src\
│   ├── data\           # Member 1
│   ├── prompts\        # Member 2
│   ├── rewards\        # Member 2
│   ├── eval\           # Member 1
│   ├── training\       # Member 3
│   ├── experiments\    # Member 3
│   └── common\         # shared Sample, parse_response, config
├── configs\            # verl YAML per run
├── scripts\            # launch_run.sh, download_data.py, build_retrieval.py
├── tests\
├── results\            # gitignored; metrics JSONs + logs
├── docs\
└── requirements.txt
```

### Dev workflow

- **Laptop:** unit tests + reward sanity checks + eval harness on fake small outputs. No GPU needed.
- **Server:** full training via `scripts/launch_run.sh <config.yaml>`. Writes to `results/<run_id>/`.
- **Sync:** one shared results folder (git-lfs vs shared mount vs rsync — team decides in Week 1).

## 6. Week-by-week plan

Gate rule: no member moves to the next week's tasks until all gate items pass. If a gate slips, the whole team addresses the blocker rather than fanning out further.

### Week 1 (Oct 11-18) — Walking skeleton + env setup

**Goal:** one tiny end-to-end run completes on server; all members have working dev loops.

**Days 1-3 (together, no fan-out):**
Clone GRACE, pin the verl commit, install Python 3.12 + CUDA 12.4 on server, download HotpotQA, hand-pick 50 samples, write a hardcoded R0 reward, run 1 GRPO step on Qwen3-0.6B with gold passages inserted directly as retrieval. Pair-debug every failure together.

**Days 4-7 (first fan-out):**

| Member 1 | Member 2 | Member 3 |
|---|---|---|
| Scaffold `src/data/`; HotpotQA raw loader; `Sample` dataclass in `src/common/` | Scaffold `src/prompts/` + `src/rewards/`; formalize R0 as `r0_grace()` with 3 unit tests | Scaffold `src/training/`; move walking-skeleton into real `scripts/launch_run.sh` + `configs/` YAML |
| Draft `parse_response()` in `src/common/` | Write `tests/test_r0.py` covering correct-evidence, correct-llm, wrong-path, invalid-format | Set up logging (W&B or local); results dir convention `results/<run_id>/` |

**Week 1 gate (all must pass):**
- [ ] `pytest tests/` green on each member's laptop
- [ ] 1-step training run completes on server using `launch_run.sh`
- [ ] Shared `Sample` + `parse_response` imported by code in all three lanes
- [ ] Results-sync mechanism agreed and documented in `docs/`

### Week 2 (Oct 19-25) — Full HotpotQA + R0 reproduction

**Goal:** full-size R0 + GRPO run on HotpotQA produces Balanced Accuracy within 10 points of GRACE's published number. Without this, R1/R2 results aren't interpretable.

| Member 1 | Member 2 | Member 3 |
|---|---|---|
| BM25 (`rank-bm25` or `pyserini`) + Qwen3-Embedding-0.6B dense retriever; balanced ans/unans split (target 296:204 for HotpotQA) | R0 production-quality + exhaustive unit tests covering all branches + edge cases (misquote, zero-overlap evidence, question-copy) | Full-size GRPO config; launch first full R0 run; wire checkpoint saving |
| Cache retrievals to `data/cache/` so training doesn't recompute | Start sketching R1 (design in `docs/rewards.md`) | Build `src/eval/evaluate.py`: `evaluate.py --checkpoint X --split Y → metrics.json` with EM + BERTScore (LJ added W3) |

**Week 2 gate:**
- [ ] Retrieval cache built; split statistics documented in `docs/data.md`
- [ ] R0 unit tests cover all 4 formula branches
- [ ] Full R0+GRPO run completes end-to-end without manual intervention
- [ ] R0+GRPO Balanced Accuracy within 10 points of GRACE's HotpotQA number (81%); if wildly off, team halts and debugs together before W3

### Week 3 (Oct 26 – Nov 1) — Reward variants + Groq LJ + DAPO

**Goal:** R1 + R2 implemented and tested; DAPO config working; LJ metric integrated and sanity-checked.

| Member 1 | Member 2 | Member 3 |
|---|---|---|
| `src/eval/judge.py`: Groq Llama 3.3 70B client, on-disk JSON cache keyed by (prompt_hash, response_hash), low temp + fixed seed | Implement R1 (cost-sensitive) + R2 (hard-gated) with unit tests mirroring R0 coverage | DAPO verl config; launch R0+DAPO run (RQ2 half) |
| Hand-label 50 examples for LJ validation (LJ should agree with EM at ≥85%) | Reward-hacking audit utilities: per-batch path-share, evidence-length distribution, question-copy fraction | `scripts/run_grid.sh` — enqueues (reward × optimizer) pairs from a YAML grid file |
| Add LJ to `evaluate.py`; re-eval the W2 R0 checkpoint with LJ | | |

**Week 3 gate:**
- [ ] R1 and R2 pass unit tests with same coverage as R0
- [ ] LJ agrees with EM at ≥85% on the 50-example validation set
- [ ] R0+DAPO run completes with metrics
- [ ] `run_grid.sh` can launch the full grid with one command

### Week 4 (Nov 2-8) — Full grid execution + first analysis

**Goal:** all 4 "must" runs complete; first comparative table with preliminary RQ1/RQ2 signal.

| Member 1 | Member 2 | Member 3 |
|---|---|---|
| Result-aggregation script → markdown + CSV comparison table | Run reward-hacking audits on all completed runs; flag exploitation patterns; iterate R1/R2 only if audits show real problems | Launch + babysit the full grid; re-launch any failed run; archive all checkpoints |
| Begin QASPER data prep (stretch): raw loader, balanced split | Draft "Reward Design" section of report | Draft "Training Setup" section of report |

**Week 4 gate:**
- [ ] All 4 "must" runs have `metrics.json`
- [ ] Comparison table includes Balanced Accuracy, answerable EM, unanswerable abstention, LJ, per-run reward-hacking audit summary
- [ ] Preliminary answer visible for RQ1 (R2 vs R0 on unanswerable subset) and RQ2 (R0+DAPO vs R0+GRPO sample efficiency)

### Week 5 (Nov 9-15) — Should runs + hero run + report draft (= target completion)

**Goal:** full 6-run grid complete; Qwen3-1.7B hero run on winning combo; report skeleton with draft sections.

| Member 1 | Member 2 | Member 3 |
|---|---|---|
| QASPER cross-domain run on best (reward, optimizer) if time permits | Finalize reward-hacking analysis; cherry-pick illustrative failure cases from each reward | Launch R1+DAPO and R2+DAPO; Qwen3-1.7B hero run on winning combo |
| Final evaluation tables + per-dataset breakdown figures | Report: "Reward Design" + "Reward-Hacking Audit" sections complete | Report: "Training & Optimizer Comparison" section complete |

**Week 5 gate (= target completion):**
- [ ] 6-run grid in one table
- [ ] Qwen3-1.7B hero run metrics available (or explicit decision not to run)
- [ ] Report has drafts of: Abstract, Background, Method, Results, Discussion

### Week 6 (Nov 16-22) — Buffer / stretch / ablations

If gates all passed on time: hyperparameter ablations on winning reward (α, β, γ, λ), full QASPER run, error analysis with ~20 failure cases per reward classified by failure type.
If gates slipped: catch-up on whatever is behind.

### Week 7 (Nov 23-29) — Final polish + submit

Report polish, figures, citations, tables. Dry-run presentation if required. Submit before end of November.

## 7. Per-member workload summary (whole project)

| Member 1 — Data / Retrieval / Eval infra | Member 2 — Reward design & analysis | Member 3 — RL training & experimentation |
|---|---|---|
| HotpotQA loader + balanced split | R0/R1/R2 implementation + unit tests | verl + vLLM env setup |
| BM25 + dense retriever + caching | Prompt templates | GRPO + DAPO configs |
| EM + BERTScore + Groq LJ + cache | Reward-hacking audit utilities | Launch + babysit all training runs |
| LJ 50-example validation set | Report: reward design + audit sections | Grid runner + hero run |
| QASPER cross-domain (stretch) | Design-doc any new reward tweaks | Report: training + optimizer sections |
| Result aggregation + tables | | Checkpoint + log management |

Everyone owns their lane's tests and the report section corresponding to their work.

## 8. Risks and fallbacks

| Risk | Likelihood | Impact | Fallback |
|---|---|---|---|
| verl install / GPU driver issues on week 1 | High | Blocks everything | Walking-skeleton pair-debug addresses this. If still broken by day 3, fall back to vanilla TRL + GRPO loss (lose DAPO comparison but keep RQ1). |
| R0 Balanced Accuracy wildly off GRACE in W2 | Medium | Blocks R1/R2 interpretation | Hard gate forces whole-team halt. Check: prompt format, retrieval ans/unans ratio, LoRA rank, model match. Up to 1 week of W3 allocated before documenting the gap. |
| R2 collapses the policy to always-abstain | Medium | Expected failure mode — informative | Document as a positive RQ1 result (strict gating too strong). Add R1.5 = R2 with soft gate boundaries. |
| DAPO config instability | Medium | Weakens RQ2 answer | Use verl's default DAPO hyperparameters from the GRACE recipe verbatim. Don't tune — tuning differences will swamp the GRPO/DAPO signal. |
| Groq rate limit hit mid-eval | Low | Eval delays | On-disk cache + sleep-and-retry on 429. Fallback to local Qwen3-4B zero-shot judge. |
| One member unavailable for a week | Medium | Lane stalls | Shared `Sample` + `parse_response` contract means another member can pick up the lane. Member 3 can run Member 1's eval; Member 1 can run Member 2's audits. |
| Qwen3-0.6B ceiling too low to show reward differences | Low-Medium | Weak final signal | Week 5 Qwen3-1.7B hero run exists for exactly this. Keep the grid cheap enough to afford one bigger run. |
| Dataset format drift | Low | Day-1 blocker | Pin HuggingFace datasets revision hash. |

## 9. Success criteria

- **Minimum viable (must, by W5):** R0 reproduction within 10 points of GRACE; R0/R1/R2 × GRPO on HotpotQA; one comparison table; RQ1 answered with evidence.
- **Full success (target, by W5):** Minimum viable + full 6-run grid + reward-hacking audit per run + RQ2 answered via GRPO vs DAPO sample-efficiency curves + Qwen3-1.7B hero run.
- **Stretch (W6-7):** QASPER cross-domain run + hyperparameter ablation on winning reward + error-case classification.

### What "answered RQ1/RQ2" looks like

- **RQ1 answered** when Balanced Accuracy and unanswerable abstention accuracy for R0/R1/R2 are plotted side by side. A monotonic improvement R0 → R1 → R2, or a specific inversion at R2, is the result.
- **RQ2 answered** when GRPO and DAPO sample-efficiency curves (Balanced Accuracy vs training step) are overlaid for the same reward. The faster-converging / higher-plateau optimizer is the answer.

## 10. Non-goals (explicit)

- Training a new reward model.
- Fine-tuning retrievers.
- Multi-GPU distributed training (single-GPU LoRA is enough for Qwen3-0.6B/1.7B).
- Novel RL algorithms (use verl's existing GRPO/DAPO recipes verbatim).
- Human eval beyond the 50-example LJ validation set.

## 11. Next step

Hand this spec to the superpowers:writing-plans skill to produce a concrete, task-level implementation plan the team can execute against.
