# Baseline reward analysis — 2026-10-21

**Inputs:** 50 Qwen3-0.6B pre-RL predictions from `results/local_sanity/20261020-194948/predictions.jsonl` (see `docs/log/2026-10-20-local-sanity.md` for how those were produced), joined back to `data/raw/hotpot_tiny.jsonl` for gold fields.

**Command:**
```
.venv/Scripts/python.exe scripts/score_predictions.py \
    results/local_sanity/20261020-194948/predictions.jsonl \
    data/raw/hotpot_tiny.jsonl \
    --out results/local_sanity/20261020-194948/reward_analysis
```

**Outputs:** `results/local_sanity/20261020-194948/reward_analysis/{rewards.json, rewards_per_sample.jsonl, reward_hist.png}` (gitignored under `results/`).

---

## Headline numbers

| Reward | n | mean | std | min | max | range (discrimination) |
|---|---:|---:|---:|---:|---:|---:|
| R0 (GRACE control) | 50 | **+1.078** | 0.733 | 0.000 | 2.000 | 2.00 |
| R1 (cost-sensitive) | 50 | **+0.998** | 0.695 | 0.000 | 2.000 | 2.00 |
| R2 (hard-gated) | 50 | **+0.224** | 0.223 | 0.000 | 0.700 | 0.70 |

## Breakdown by (is_answerable, path)

(The tiny set is fully answerable, so only `is_answerable=True` buckets appear.)

| Bucket | n | R0 mean | R1 mean | R2 mean |
|---|---:|---:|---:|---:|
| answerable, `evidence` path | 34 | +1.570 | +1.453 | **+0.330** |
| answerable, `llm` path | 1 | +0.500 | +0.500 | +0.000 |
| answerable, `invalid` path | 15 | +0.000 | +0.000 | +0.000 |

## What this confirms

1. **All three reward functions produce non-degenerate distributions on real outputs.** Unit tests confirmed branch correctness; this is the first time the rewards have seen free-form generation and they still produce the expected ordering (`evidence` > `llm` > `invalid`).

2. **R2 is 3–5× stricter than R0/R1 on the same inputs — by design.** The multiplicative gate collapses most outputs: even an evidence-path response averages only +0.330 under R2 vs +1.570 under R0, because R2 further gates on verbatim-quoted evidence (Rformat ∈ {0, 1}). This is exactly the RQ1 extreme the plan calls for.

3. **R1 is slightly stricter than R0 (−0.07 on the full set, −0.12 on the evidence bucket).** This is the quote-containment indicator 𝟙[quote ⊂ C_ret] doing its job — penalizing responses that produce ROUGE-matching evidence that didn't actually come from the retrieved context.

4. **Invalid-format outputs correctly score 0 under all three rewards.** The 30% invalid-format rate noted in `docs/log/2026-10-20-local-sanity.md` translates directly to 15/50 = 30% of training signal at value 0 — meaning for a GRPO policy update, roughly 30% of rollouts contribute no gradient. RL training's first measurable job is reducing this fraction.

5. **The one `llm`-path response scores R0=R1=0.5, R2=0.0.** For R0/R1 that's `R_f` only (format valid, path wrong because the sample is answerable). For R2 that's `R_path=0` collapsing the whole product to 0 — again, R2 behaving as designed.

## Implications for Week-3 server runs

- The gap between R0 and R2 reward values means **absolute reward numbers from GRPO logs will not be directly comparable across rewards**. The plan's primary metric is already **Balanced Accuracy on the eval set**, not training reward — this run confirms we were right to pin to it.
- The R2 maximum observed on this baseline (0.70) is well below its theoretical max (1.0). R2's ceiling will be lifted only when the policy learns verbatim quoting + correct path jointly. If R2 training reward stays below ~0.1 after a few hundred steps, that's the "collapsed to always-abstain" failure mode the plan predicts — we'd document it as a positive RQ1 result and add R1.5 with soft gates.
- The 1/50 `llm`-path rate on a fully-answerable split is **not** a bug — it's the base model's tendency on answerable queries. The real test is what happens on the 500-sample balanced val split (~50% unanswerable), where the correct `llm`-path rate should rise sharply after training.

## Reproducing

```bash
# Requires the sanity run to have been done first (see docs/log/2026-10-20-local-sanity.md)
python scripts/score_predictions.py \
    results/local_sanity/<timestamp>/predictions.jsonl \
    data/raw/hotpot_tiny.jsonl \
    --out results/local_sanity/<timestamp>/reward_analysis
```

Runs in ~2 seconds on CPU; no model loading. Will work identically on any trained-checkpoint `predictions.jsonl` once Week-2 Gate runs land, so this becomes the standard reward-audit workflow for every run.
