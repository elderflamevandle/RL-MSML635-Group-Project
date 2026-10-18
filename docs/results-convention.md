# Results convention

Every training run writes to `results/<run_id>/` where:

- `<run_id>` = `<reward>_<optimizer>_<model>_s<seed>_<YYYYMMDD-HHMMSS>` (built by `src/training/run_naming.py`)
- `config.yaml` — copy of the launched config
- `train.log` — stdout/stderr of the training command
- `checkpoints/` — model checkpoints (gitignored)
- `rollouts/` — sampled rollouts per step (gitignored, large)
- `metrics.json` — final evaluation metrics (added by Member 1's `evaluate.py`)
- `audit.json` — reward-hacking audit (added by Member 2's audit script)
- `predictions.jsonl` — per-sample predictions used for eval + audit

Sync: see `docs/team.md` for the agreed mechanism (NFS mount vs rsync vs git-lfs).
