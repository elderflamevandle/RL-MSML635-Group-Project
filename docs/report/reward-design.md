# Reward Design + Reward-Hacking Audit

**Section owner:** Member 2
**Depends on:** Week 3 Task 3.2 (R1/R2/audit code; done) + Week 4 Task 4.3 (audits on real runs; pending server).

## Overview

GRACE's reward R₀ is a sum of four conditional branches. We hypothesised that
*summing* instead of *gating* is the dominant failure mode in R₀: a response
that is well-formatted but ungrounded still earns the format bonus `R_f`, and
there is no asymmetric cost for hallucinating on unanswerable queries. R₁
tightens R₀ with an explicit hallucination penalty and a quote-containment
gate on `R_f`. R₂ pushes the hypothesis further: all terms gate
multiplicatively, so any single failure collapses the reward to 0.

## R0 — GRACE baseline

Formulation (from `src/rewards/r0_grace.py`):

```
R0 = 0,                                   invalid format
   = R_f,                                 valid format, wrong path
   = R_f + R_p,                           correct <llm> path
   = R_f + R_p + α·R_Lev + β·R_Lans,      correct <evidence> path
```

with `R_f = 0.5`, `R_p = 1.0`, `α = β = 0.5` (`R_Lev` / `R_Lans` are ROUGE-L
against gold evidence / gold answer). This matches GRACE's published values
and is intentionally frozen as the baseline against which R₁ and R₂ are
evaluated.

**Known weaknesses R₁/R₂ attack:**

1. **Ceiling asymmetry.** The `<evidence>` ceiling (`R_f + R_p + α + β` = 2.5)
   exceeds the `<llm>` ceiling (`R_f + R_p` = 1.5) → systemic bias toward
   answering, even when correct behavior is abstention.
2. **No cost for hallucination.** A policy that answers an unanswerable query
   with a confident wrong answer loses only `R_p` relative to the correct
   `<llm>` path. No negative signal.
3. **Nothing checks that quoted evidence actually came from the retrieved
   context.** ROUGE-L against gold rewards fabrication that happens to
   phrase-match gold.

## R1 — Cost-sensitive with hallucination penalty

Formulation (from `src/rewards/r1_cost_sensitive.py`):

```
R1 = -λ,                                              hallucinated (answered on unanswerable)
   = 0,                                                invalid format
   = R_f,                                              valid format, wrong path (non-hallucination)
   = R_f + R_p + γ·R_Lans,                             correct <llm> path  (closes ceiling asymmetry)
   = R_f·𝟙[quote ⊂ C_ret] + R_p + α·R_Lev + β·R_Lans,  correct <evidence> path (quote-containment gate)
```

Starting values: `λ = 1.0`, `γ = 0.5`. α and β inherited from R₀.

**Three fixes layered on R₀, each independently testable in audit:**

- `-λ` on hallucination directly penalises guessing on unanswerable.
- `γ · R_Lans` on `<llm>` path closes the ceiling asymmetry (now `<llm>` can
  reach `R_f + R_p + γ` = 2.0 vs `<evidence>` ceiling 2.5).
- `𝟙[quote ⊂ C_ret]` zeroes the format bonus on fabricated evidence while
  preserving the content bonus — so a response with the right answer but a
  fabricated quote keeps 1.5 of its 2.5 ceiling, losing exactly `R_f`.

## R2 — Hard-gated multiplicative

Formulation (from `src/rewards/r2_hard_gated.py`):

```
R2 = R_format · (w_p · R_path + w_q · R_path · R_quality)

R_format  ∈ {0, 1}   — valid tags AND verbatim-quoted evidence
R_path    ∈ {0, 1}   — correct <evidence>/<llm> choice
R_quality ∈ [0, 1]   — evidence path: 0.5·R_Lev + 0.5·R_Lans
                     — llm path:      R_Lans (disclaimer quality)
```

Starting values: `w_p = 0.4`, `w_q = 0.6`. Everything gates multiplicatively,
so any single failing gate collapses R₂ to 0.

**Design intent for RQ1:** this is the extreme test. If R₂ wins on Balanced
Accuracy, hard gates are sufficient and R₁'s soft shaping is unnecessary.
If R₂ collapses the policy to always-abstain, hard gates are too strict and
R₁ is the sweet spot. Either result is informative.

## Baseline reward audit on pre-RL predictions

Before any training, we scored all three reward functions on the 50
Qwen3-0.6B baseline predictions from `results/local_sanity/20261020-194948/`
(see `docs/log/2026-10-21-baseline-reward-analysis.md` for the full run).
This answers "do our reward functions distinguish real outputs?" before we
spend compute on RL.

| Reward | mean | std | min | max | discrimination range |
|---|---:|---:|---:|---:|---:|
| **R0** | +1.078 | 0.733 | 0.000 | 2.000 | 2.00 |
| **R1** | +0.998 | 0.695 | 0.000 | 2.000 | 2.00 |
| **R2** | +0.224 | 0.223 | 0.000 | 0.700 | 0.70 |

Breakdown by (answerable, path):

| Bucket | n | R0 | R1 | R2 |
|---|---:|---:|---:|---:|
| ans, `evidence` | 34 | +1.570 | +1.453 | +0.330 |
| ans, `llm` | 1 | +0.500 | +0.500 | +0.000 |
| ans, `invalid` | 15 | +0.000 | +0.000 | +0.000 |

**Three confirmations:**

- R₁ is slightly stricter than R₀ (`+1.453` vs `+1.570` on evidence-path) —
  the quote-containment gate is actively zeroing `R_f` on a non-trivial
  fraction of responses. This is the gate working as designed.
- R₂ is 3–5× stricter on average and never exceeds 0.70 (vs theoretical max
  1.0) — multiplicative gating collapses most outputs. R₂ will see almost
  no positive signal on an untrained policy.
- All three rewards correctly return 0.0 on invalid-format outputs. 15/50 =
  30% of training rollouts will contribute no gradient signal from the base
  policy; reducing this is RL's first measurable task.

## Baseline failure-mode classification

Running the full classifier (`scripts/classify_failures.py`) on the same
50 predictions gives the breakdown R₁/R₂ are designed to attack:

| Failure category | n | % | Reward attacking this |
|---|---:|---:|---|
| `wrong_content` (grounded quote, wrong answer) | 19 | 38% | all; needs model capacity |
| `format_invalid` | 15 | 30% | all (zero reward) |
| `fabricated_evidence` (quote not in retrieved context) | 10 | 20% | **R₁ (quote-containment gate), R₂ (R_format=0)** |
| `answer_format` (semantic match, EM miss) | 3 | 6% | will show up in BERTScore / LJ, not EM |
| `correct` (EM match) | 2 | 4% | — |
| `wrong_path` (llm on answerable) | 1 | 2% | R₁ (if later unanswerable) |

**Conditional rates:**

- `fabricated_evidence_rate_on_evidence_path_answerable` = **29.4%**
- `abstention_rate_on_answerable` = 2.0%
- `hallucination_rate_on_unanswerable` = 0.0% (tiny set is all-answerable)

That 29% fabrication rate is the single most compelling piece of evidence
for R₁'s design. If R₁ training drives it below ~5% while R₀ leaves it
unchanged, we have a clean positive result for the quote-containment gate.

## Expected post-training signatures (hypotheses)

To be filled in from `scripts/score_predictions.py` + `scripts/classify_failures.py`
run on each trained checkpoint's `predictions.jsonl`.

| Signal | R0 (prior) | R1 (prior) | R2 (prior) |
|---|---|---|---|
| `fabricated_evidence_rate` | high → modest ↓ | high → sharp ↓ (gated) | high → ~0 (collapses R₂) |
| `path_shares.invalid` | 30% → low | 30% → low | 30% → low |
| `path_shares.llm` on balanced val | low → middling | low → middling / high | low → very high (abstain) |
| `balanced_accuracy` | 0.04 → ~0.81 (GRACE claim) | 0.04 → ≥ R₀ (expected) | 0.04 → ≥ R₀ or collapses |

## Audit summary (across R0/R1/R2 × GRPO/DAPO)

To be filled after Week 4 "must" runs land. Produced by:

```
python scripts/score_predictions.py results/<run>/predictions.jsonl data/cache/dense_val.jsonl --out results/<run>/reward_analysis
python scripts/classify_failures.py results/<run>/predictions.jsonl data/cache/dense_val.jsonl --out results/<run>/failures
```

| Run | fabricated_quote_rate | abstention_on_ans | path (ev / llm / inv) | mean R₀ |
|---|---|---|---|---|
| r0_grpo | | | | |
| r1_grpo | | | | |
| r2_grpo | | | | |
| r0_dapo | | | | |
| r1_dapo | | | | |
| r2_dapo | | | | |

### Red flags to catch (automated)

- `path_shares.evidence > 0.95` → policy collapse (over-answered)
- `path_shares.llm > 0.95` → policy collapse (over-abstained; expected R₂ failure mode)
- `fabricated_evidence_rate > 0.1` → quote fabrication still unlearned
- `wrong_content_rate > 0.3` → model capacity, not reward, is the bottleneck
- `evidence_length_mean_tokens > 50` → policy padding evidence to game `R_Lev`

## Takeaway (to be finalised post-grid)

Preliminary (baseline-only): all three reward functions behave as designed on
free-form outputs, with the exact ordering R₀ > R₁ > R₂ in strictness. The
baseline has a 30% invalid-format rate and 29% fabrication rate on
evidence-path answerable queries, so R₁'s quote-containment gate has a large
measurable target to attack. Final verdict pending Week 4 grid.
