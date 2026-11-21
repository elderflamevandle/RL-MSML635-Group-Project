# Training & Optimizer Comparison

**Section owner:** Member 3
**Depends on:** Tasks 4.1 (must-have grid) + 5.1 (should-have grid + hero run).

## Setup

- **Base policy:** Qwen3-0.6B with LoRA, rank `r = 16`, α = 32 (match GRACE).
- **Hero policy (Week 5):** Qwen3-1.7B with the same LoRA config on the
  winning (reward, optimizer).
- **Framework:** `verl` pinned to commit `4aa02fe` (GRACE paper's reference).
- **Rollout backend:** vLLM, `gpu_memory_utilization = 0.6`,
  temperature 0.9 at rollout time, 0.0 at eval time.
- **GRPO specifics:** `n = 8` rollouts per group; `kl_coef = 0.001`;
  `advantage_normalization = true`.
- **DAPO specifics:** decoupled clip (`clip_ratio_low = 0.2`,
  `clip_ratio_high = 0.28`), `dynamic_sampling = true`.
- **Optimizer:** AdamW, `lr = 1e-5`, `ppo_mini_batch_size = 16`,
  `train_batch_size = 32`, cosine schedule, 400 total training steps,
  checkpoints every 100.
- **Compute budget:** single seed per combination (compute-constrained;
  see Limitations in REPORT.md). Six "must + should" runs + one 1.7B hero =
  seven total GPU-days on a single A-class card.

Configuration files are all committed under `configs/` and lint-validated
with `scripts/preflight.py` before launch.

## Data

Trained on `data/cache/dense_train.jsonl` (2000 HotpotQA samples, balanced
~51% answerable / 49% unanswerable via distractor-swap). Validation on
`data/cache/dense_val.jsonl` (500 samples, same balance). Both files emit
a `prompt` field so verl's `prompt_key: prompt` reads directly.

## Reward comparison (RQ1)

**All GRPO**, reward swapped. Fills from `results/<run>/metrics.json` via
`scripts/make_comparison_table.py`.

| Metric | R0+GRPO | R1+GRPO | R2+GRPO |
|---|---|---|---|
| balanced_accuracy | | | |
| em_answerable | | | |
| abstention_unanswerable | | | |
| bertscore_f1_answerable | | | |
| judge_score_answerable (LJ) | | | |
| fabricated_evidence_rate | | | |
| wall time (h) | | | |

**Hypothesis to confirm or reject (RQ1):**
R1 should improve `abstention_unanswerable` and `balanced_accuracy` vs R0
without a steep EM drop (soft shaping). R2 should push abstention
further but risk EM collapse (hard gating). The winner on `balanced_accuracy`
answers RQ1; the direction of the tradeoff informs reward-design guidance
beyond this project.

## Optimizer comparison (RQ2)

R0 fixed; GRPO vs DAPO head-to-head.

| Metric | R0+GRPO | R0+DAPO | Δ |
|---|---|---|---|
| balanced_accuracy | | | |
| em_answerable | | | |
| abstention_unanswerable | | | |
| steps to plateau | | | |
| wall time (h) | | | |
| kl_divergence @ final step | | | |

**Hypothesis (RQ2):** DAPO's decoupled clip + dynamic sampling should give
either (a) higher final `balanced_accuracy` or (b) faster convergence to
R₀+GRPO's plateau. If both metrics are inside noise, GRPO's simplicity wins
by default (fewer hyperparameters to document for a 2-week reproducer).

## Hero run (Week 5)

- Config: `configs/hero_1.7b.yaml` (not yet created — will mirror the
  winning base-model config with `path: Qwen/Qwen3-1.7B`,
  `total_training_steps: 800`, and reduced `train_batch_size` if VRAM
  demands).
- Qwen3-1.7B weights pre-staged in local HF cache (`4.08 GB`) during
  Week 2 to shave download time off Week 5.

| Metric | best 0.6B | 1.7B hero | Δ |
|---|---|---|---|
| balanced_accuracy | | | |
| em_answerable | | | |
| bertscore_f1_answerable | | | |

## Training-time observations

To be filled from the per-run training log (`results/<run>/train.log`) during
Week 4 debrief. Specifically we want:

- Rollout path-share over training (plot via `scripts/plot_path_shares.py`
  after augmenting evaluate.py to emit intermediate metrics).
- `kl_divergence` trajectory — does R₂ induce catastrophic drift?
- OOM / crash notes (link to `docs/log/<date>-grid-<run>.md`).
- Any config edits forced by actual verl errors on the pinned commit
  (expected area for Week-3 pair-debugging — current configs are
  best-guesses from GRACE's reference commit).

## Takeaway

To be finalised after full grid. Preliminary pass criterion from the plan:
R₀+GRPO must reproduce GRACE's HotpotQA Balanced Accuracy to within 10
points (0.71 ≤ BA ≤ 0.91). Without that gate pass, R₁/R₂ comparisons are
not interpretable.
