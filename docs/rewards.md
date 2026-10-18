# Reward design

See `docs/superpowers/specs/2026-10-11-rl-rag-reward-design-plan.md` §4 for background and research framing. This file is the implementation-level reference.

## R0 — GRACE baseline (control)

Implemented in `src/rewards/r0_grace.py`. Config: `R0Config(Rf=0.5, Rp=1.0, alpha=0.5, beta=0.5)`.

Formula:

```
R0 = 0,                                    invalid format
     Rf,                                   valid format, wrong path
     Rf + Rp,                              correct <llm> path
     Rf + Rp + α·RLev + β·RLans,           correct <evidence> path
```

**Known weaknesses that R1/R2 attack:**
1. **Ceiling asymmetry.** Max `<evidence>` reward = Rf+Rp+α+β = 2.5; max `<llm>` reward = Rf+Rp = 1.5. The policy is systemically biased toward answering (hallucination risk).
2. **No hallucination penalty.** Answering on an unanswerable sample scores Rf (wrong path, same as any other wrong-path mistake). Not costly enough.
3. **No quote-fabrication detection.** `RLev` ROUGE-matches gold evidence; nothing checks the quote actually came from the retrieved passages. The model can invent a plausible quote.

## R1 — Cost-sensitive reward with hallucination penalty (Week 3)

Will be implemented in `src/rewards/r1_cost_sensitive.py`. Config: `R1Config(Rf=0.5, Rp=1.0, alpha=0.5, beta=0.5, gamma=0.5, lam=1.0)`.

Formula:

```
R1 = -λ,                                              hallucinated (answered on unanswerable)
      0,                                              invalid format
      Rf,                                             valid format, wrong path (non-hallucination)
      Rf + Rp + γ·RLans,                              correct <llm> path  (quality bonus)
      Rf·𝟙[quote ⊂ Cret] + Rp + α·RLev + β·RLans,   correct <evidence> path (quote-containment gate on Rf)
```

Semantics:
- "Hallucinated" = `parsed.path == "evidence" AND not sample.is_answerable`.
- "quote ⊂ Cret" = every quoted evidence string is a substring of at least one `sample.retrieved_passages` entry (uses `src.rewards.metrics.quote_in_passages`).
- γ·RLans on `<llm>` path closes the ceiling-asymmetry weakness of R0.
- `-λ` penalty directly addresses the dominant failure mode.

## R2 — Hard-gated multiplicative reward (Week 3, RQ1 extreme test)

Will be implemented in `src/rewards/r2_hard_gated.py`. Config: `R2Config(w_p=0.4, w_q=0.6)`.

Formula:

```
R2 = Rformat × (w_p·Rpath + w_q · Rpath · Rquality)
```

With components in {0,1} or [0,1]:
- `Rformat ∈ {0,1}` — valid tags AND (if evidence path) every quote is a substring of retrieved passages.
- `Rpath ∈ {0,1}` — correct `<evidence>`/`<llm>` choice given `is_answerable`.
- `Rquality ∈ [0,1]`:
  - evidence path: `0.5·RLev + 0.5·RLans`
  - llm path:      `RLans` (quality of disclaimer vs reference)

Every upstream gate failing collapses R2 to 0 — no partial credit for "right path, wrong evidence."

**Interpretation:** if R2 wins on Balanced Accuracy (especially on the unanswerable subset), gating works and RQ1 is answered positively. If R2 collapses the policy to always-abstain, gating is too strict and R1 is the sweet spot. Either outcome answers RQ1.

## Reward-hacking audit (Week 3-4)

Per-batch stats computed by `src/rewards/audit.py` on rollouts:
- share of responses on each path (`evidence` / `llm` / `invalid`)
- mean + histogram of quoted-evidence length (in tokens)
- fraction of answers that contain the question verbatim
- fraction of evidence quotes NOT found in `retrieved_passages`

Alarm thresholds (see Week 4 Task 4.3 in the plan):
- path_shares with `evidence > 0.95` or `llm > 0.95` → policy collapse
- `fabricated_quote_rate > 0.1` → fabrication exploitation
- `question_copy_rate > 0.1` → reward hacking via copy
- `evidence_length_mean_tokens > 50` → over-quoting to inflate RLev

## Hyperparameter starting values and tuning policy

All starting values come from GRACE (for R0 inherited parameters) or principled defaults (for R1/R2 new parameters). **Tuning policy**: no hyperparameter sweeps until Week 6. Week 2-5 use these defaults verbatim so that reward comparisons are not confounded with tuning variance.

| Parameter | R0 | R1 | R2 | Source |
|---|---|---|---|---|
| Rf | 0.5 | 0.5 | — | GRACE |
| Rp | 1.0 | 1.0 | — | GRACE |
| α | 0.5 | 0.5 | — | GRACE |
| β | 0.5 | 0.5 | — | GRACE |
| γ | — | 0.5 | — | Chosen to roughly balance `<llm>` max (1.5+γ=2.0) with `<evidence>` max (2.5) — closes but does not eliminate the asymmetry |
| λ | — | 1.0 | — | One full "format bonus" unit of penalty; same scale as the reward side |
| w_p | — | — | 0.4 | Chosen so path-correct-but-quality-zero scores 0.4 (clear reward signal) |
| w_q | — | — | 0.6 | w_p + w_q = 1.0 so the max R2 is bounded in [0,1] |
