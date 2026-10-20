# Skeleton run — <YYYY-MM-DD>

**Members present:** Member 1, Member 2, Member 3
**GPU server:** <hostname>
**Driver:** `nvidia-smi` CUDA version <X.Y>
**Starting commit:** <sha>

## Env setup

- [ ] `bash scripts/setup_env.sh` completed
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True`
- [ ] `python -c "import vllm, verl; print(vllm.__version__, 'verl ok')"` passes

Fix any failures by:
- CUDA mismatch → pick vllm wheel matching driver CUDA
- verl build fail → pip install -v and read the compiler error
- HF cache → confirm HF_TOKEN env + ~/.cache/huggingface writable

## Walking skeleton

Launch: `bash scripts/launch_run.sh configs/skeleton.yaml`

### Iteration log

| Attempt | Change | Error (if any) | Resolution |
|---|---|---|---|
| 1 | initial config | <paste first few lines of error> | <what you changed> |
| 2 | | | |

### Final working config

- `_meta.reward`:
- `_meta.optimizer`:
- `model.path`:
- `train_batch_size`:
- `gpu_memory_utilization`:
- Any renamed/removed fields:

### Rollout sample (first step)

```
<paste 1-2 example model outputs so future-you can tell if the base model is in RAG format at all>
```

### Metrics observed

- loss (step 1):
- lr:
- rollouts/step:
- step wall time:

## Checkpoint

- [ ] All 3 members confirmed training step completed without crash
- [ ] Working config committed as `configs/skeleton.yaml` on `main`
- [ ] Ready to fan out into Member-1/2/3 lanes for Week 2
