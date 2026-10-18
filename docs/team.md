# Team Logistics — MSML635 Group 5

**Status:** draft skeleton — fill in the TODOs on Day 1.

## Member assignments

Initial assignment (swappable mid-project if load gets uneven):

| Member | Name | Lane |
|---|---|---|
| Member 1 (Data) | TODO | HotpotQA loader, retrievers, balanced split, cache; Week 2 full retrieval build; QASPER stretch (W5) |
| Member 2 (Rewards) | TODO | R0 hardening, R1 cost-sensitive, R2 hard-gated, audit utilities; reward-design report section |
| Member 3 (Training) | TODO | verl env, launchers, GRPO/DAPO configs, grid runner, hero run; training comparison report section |

## GPU server

- **Hostname / address:** TODO
- **SSH command:** TODO (e.g., `ssh <user>@<host> -p <port>`)
- **Who has access:** TODO
- **Workspace path on server:** TODO (recommend `~/rl-rag/`)
- **Shared venv path:** `~/rl-rag-venv` (per `scripts/launch_run.sh`)

## Groq API key

- **Account owner:** TODO (one member creates the account)
- **Key shared via:** TODO (password manager — 1Password / Bitwarden / etc.)
- **Verified with curl test (Task 0.1 Step 3):** [ ] Yes / [ ] No

## Experiment tracking

Pick one:

- [ ] W&B team account — URL: TODO
- [ ] Local tensorboard only (results synced via mechanism below)

## Results sync mechanism

Pick one (recommend NFS or rsync; `git-lfs` adds friction):

- [ ] Shared NFS mount — path: TODO
- [ ] `rsync` + cron — command: TODO, cadence: TODO
- [ ] `git-lfs` — not recommended

## Weekly sync meeting

- **Day / time:** TODO
- **Where:** TODO (Zoom link / room)
- **Standing agenda:** blockers, Week-gate status, next-week ownership

## Checkpoint

- [ ] All three members confirmed SSH access to GPU server
- [ ] Groq curl test returns `200` with a non-empty reply
- [ ] Results-sync mechanism tested with a dummy file
