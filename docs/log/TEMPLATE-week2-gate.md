# Week 2 Gate — first R0+GRPO run

**Date:** <YYYY-MM-DD>
**Owner:** Member 3
**Starting commit:** <sha>

## Config used

`configs/r0_grpo.yaml` as of commit <sha>. Key values:

- `model.path`:
- `train_batch_size`:
- `n_rollouts`:
- `total_training_steps`:
- `data.train_files` / `val_files`:

## Run

- `bash scripts/launch_run.sh configs/r0_grpo.yaml`
- Started: <HH:MM>
- Finished: <HH:MM>
- Wall time:
- Run dir: `results/<run_id>/`

## Evaluation

`python -m src.eval.evaluate --checkpoint results/<run_id>/last --split data/cache/dense_val.jsonl --out results/<run_id>/metrics.json`

### metrics.json

```json
<paste full metrics.json here>
```

### Gate check

- [ ] `balanced_accuracy` ∈ [0.71, 0.91]: **<value>**  ✅/❌
- [ ] `em_answerable` reasonable (expect ~0.55-0.7 for GRACE baseline)
- [ ] `abstention_unanswerable` ≥ 0.7 (base-model TNR floor)

### If ❌: diagnostic checklist

- [ ] Prompt matches GRACE paper (`src/prompts/templates.py:RAG_SYSTEM`)
- [ ] Format valid rate > 0.5 (if not, model never learned the tag schema → longer train, warm start, or reward shaping)
- [ ] `path_shares.evidence` not collapsed to >0.95 or <0.05 (policy collapse)
- [ ] Learning rate not too high (default 1e-6 for GRPO is typical)
- [ ] LoRA rank matches GRACE (r=16 is their default)

## Decision

- [ ] Proceed to Week 3 (Tasks 3.1, 3.2, 3.3 fan out to members)
- [ ] HALT — root cause: _______________. Re-run after fix.
