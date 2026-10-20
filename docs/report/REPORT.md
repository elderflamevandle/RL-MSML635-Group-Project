# MSML635 Group 5 — RL-Assisted RAG: Reward Design

**Authors:** Member 1, Member 2, Member 3
**Course:** MSML635
**Date:** <fill at report time>

> **Status:** scaffold only. Replace each `<...>` and section stub during Week 5 Task 5.4.

## Abstract

<One paragraph: motivation (RAG hallucinates + over-abstains), approach (compare R0/R1/R2 rewards × GRPO/DAPO on HotpotQA), result headline (which combo won on balanced accuracy; whether cross-domain QASPER held).>

## 1. Introduction

<Problem statement from the proposal. Cite GRACE (ICLR 2025, grounded response and abstention). Research questions: RQ1 (which reward best suppresses hallucination while preserving answerability?), RQ2 (GRPO vs DAPO sample efficiency on this task).>

## 2. Setup

- **Dataset:** HotpotQA distractor subset (2000 train, 500 val) per GRACE. QASPER cross-domain (Week 5).
- **Retrievers:** BM25 + Qwen3-Embedding-0.6B. k=5 per sample.
- **Balanced split:** 50/50 answerable/unanswerable via distractor-swap.
- **Model:** Qwen3-0.6B base; hero run on Qwen3-1.7B.
- **Framework:** verl @ `4aa02fe` (GRACE's pinned commit).
- **Evaluation:** EM (answerable), TNR (unanswerable), balanced accuracy, BERTScore, LLM-as-judge (Groq llama-3.3-70b-versatile, 85%+ agreement validated on 50 hand-labels).

## 3. Reward Design

See `docs/report/reward-design.md` — Member 2 fills.

- R0: GRACE baseline reward.
- R1: cost-sensitive — hallucination penalty λ, quote-containment gate on Rf.
- R2: hard-gated multiplicative — `R_format * (w_p*R_path + w_q*R_path*R_quality)`.

## 4. Training & Optimizer Comparison

See `docs/report/training.md` — Member 3 fills.

- GRPO vs DAPO on R0 (RQ2 head-to-head).
- Hero run: Qwen3-1.7B on winning (reward, optimizer).

## 5. Results

See `docs/report/results.md` — Member 1 fills; references `docs/results-table.md` auto-generated via `scripts/make_comparison_table.py`.

### 5.1 Main comparison table

<Insert generated table.>

### 5.2 RQ1 answer

<Compare `abstention_unanswerable` across R0/R1/R2 at fixed optimizer. Report winner + magnitude.>

### 5.3 RQ2 answer

<Compare R0+GRPO vs R0+DAPO on `balanced_accuracy` + sample efficiency (steps-to-plateau). Report winner.>

### 5.4 Cross-domain (QASPER)

<Does the winner on HotpotQA transfer? Report balanced accuracy drop/gain.>

## 6. Reward-Hacking Audit

See `docs/report/reward-design.md` + `docs/audit-final.md`.

- Path share collapse checks.
- Fabricated-quote rate.
- Question-copy rate.
- Evidence-length drift.

## 7. Failure Cases

See `docs/report/failure-cases.md` — 3-5 illustrative failures per reward.

## 8. Limitations

<Compute budget capped at Qwen3-1.7B hero; single seed per combo (no variance bars); HotpotQA + QASPER only.>

## 9. Conclusion

<Which reward+optimizer we recommend and why. One sentence on what would unlock the next frontier (longer context, more params, more datasets).>

## References

- Zhao et al. GRACE: Reinforcement Learning for Grounded Response and Abstention under Contextual Evidence (arXiv 2601.04525).
- Yang et al. Qwen3 Technical Report.
- Shao et al. GRPO. Yuan et al. DAPO.
- Johnson et al. HotpotQA. Dasigi et al. QASPER.
