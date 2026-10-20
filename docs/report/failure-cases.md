# Failure Cases

**Section owner:** Member 2
**Depends on:** Task 5.3 — pick 3-5 per reward from `results/<run>/predictions.jsonl`.

Each case: question + retrieved context (truncated) + model response + why it's wrong + which reward produced it.

## R0+GRPO failures

### Case R0-1 — <short label, e.g., "hallucinated date">

- **Question:** <q>
- **Gold answer:** <a>
- **Retrieved (top-k):** <trimmed>
- **Model response:** `<evidence>...</evidence><answer>...</answer>`
- **What went wrong:** <classification: hallucination / over-abstention / question copy / fabricated quote / wrong path / other>

### Case R0-2 — ...

## R1+GRPO failures

### Case R1-1 — ...

## R2+GRPO failures

### Case R2-1 — ...

## Patterns

<Across the cases, is there a dominant failure mode per reward? E.g., R0 fabricates evidence; R1 under-abstains; R2 collapses to all-llm.>
