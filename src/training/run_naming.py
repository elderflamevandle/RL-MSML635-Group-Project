"""Standardized run id. Every training run writes to results/<run_id>/."""
from datetime import datetime

def make_run_id(reward: str, optimizer: str, model: str, seed: int) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{reward}_{optimizer}_{model}_s{seed}_{ts}"
