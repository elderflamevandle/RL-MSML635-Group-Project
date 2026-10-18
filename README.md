# RL-Assisted RAG — Reward Design

MSML635 Group 5. Reward design for RL-assisted RAG, extending GRACE with cost-sensitive and hard-gated reward variants.

- Spec: `docs/superpowers/specs/2026-10-11-rl-rag-reward-design-plan.md`
- Plan: `docs/superpowers/plans/2026-10-11-rl-rag-reward-design.md`

## Setup (laptop dev)

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# .venv/bin/activate         # macOS/Linux
pip install -e ".[dev]"
pytest -q
```

Fast tests skip the slow ones (which download embedding models). To run everything:

```bash
pytest -q -m "slow or not slow"
```

## Setup (GPU server)

See `scripts/setup_env.sh` (added in Week 1 Task 1.1).
