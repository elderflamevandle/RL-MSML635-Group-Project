# Training & Optimizer Comparison

**Section owner:** Member 3
**Depends on:** Tasks 4.1 (must-have grid) + 5.1 (should-have grid + hero run).

## Setup

- Base: Qwen3-0.6B, LoRA r=16 (match GRACE).
- Hero: Qwen3-1.7B on winning (reward, optimizer).
- Verl @ `4aa02fe`.
- Single seed per combination (compute-constrained; see Limitations in REPORT.md).

## Optimizer comparison (RQ2)

R0 fixed; GRPO vs DAPO head-to-head.

| Metric | R0+GRPO | R0+DAPO | Δ |
|---|---|---|---|
| balanced_accuracy | | | |
| em_answerable | | | |
| abstention_unanswerable | | | |
| steps to plateau | | | |
| wall time | | | |

**Verdict:** <GRPO or DAPO wins; by how much; is it worth DAPO's complexity?>

## Hero run

- Config: `configs/hero_1.7b.yaml` (winning reward+optimizer, scaled model + 2× steps).
- Did scaling help?

| Metric | best 0.6B | 1.7B hero | Δ |
|---|---|---|---|
| balanced_accuracy | | | |
| em_answerable | | | |

## Observations

- Rollout path shares over training (did the model stabilize or oscillate?).
- OOM / crash notes (reference `docs/log/<date>-grid-<run>.md`).

## Takeaway

<One sentence recommendation on reward+optimizer+scale.>
