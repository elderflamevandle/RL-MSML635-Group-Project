# Team Logistics — MSML635 Group 5

**Status:** draft — items marked `[PROPOSED]` are defaults; the team should accept or edit in the kickoff call. Items marked `TODO` require a human decision before anything else moves.

## Key links

- **Repo:** <https://github.com/elderflamevandle/RL-MSML635-Group-Project>
- **Proposal (PDF):** `RL_Group_5_Proposal.pdf` in repo root
- **Project plan:** `docs/superpowers/plans/2026-10-11-rl-rag-reward-design.md`
- **Project status (always current):** `docs/STATUS.md`
- **Session handoff (Day 1):** `HANDOFF.md`
- **Pre-RL baseline:** `docs/log/2026-10-20-local-sanity.md` (Qwen3-0.6B, `BalancedAccuracy = 0.04`)

## Member assignments

Initial assignment (swappable mid-project if load gets uneven):

| Member | Name | GitHub | Primary lane | Key files to own |
|---|---|---|---|---|
| Member 1 (Data) | TODO | TODO | HotpotQA loader, retrievers, balanced split, retrieval cache; Week-2 full retrieval build; QASPER cross-domain run (W5) | `src/data/*`, `scripts/build_retrieval.py`, `scripts/download_hotpot_*.py` |
| Member 2 (Rewards) | TODO | TODO | R0 hardening, R1 cost-sensitive, R2 hard-gated, reward-hacking audits; reward-design report section | `src/rewards/*`, `scripts/run_audits.py`, `docs/rewards.md` |
| Member 3 (Training) | TODO (@elderflamevandle?) | elderflamevandle | verl env setup, launchers, GRPO/DAPO configs, grid runner, hero run; training-comparison report section | `scripts/setup_env.sh`, `scripts/launch_run.sh`, `configs/*.yaml`, `src/training/*` |

> ⚠ **Decide at kickoff:** which human takes Members 1 / 2 / 3 — the lanes above are starting points, not fixed. Members 1 and 2 can work in parallel on laptops; Member 3 needs server access first.

## Communication

- **Primary channel:** `[PROPOSED]` Discord server or Signal group — fast, async, free. Pick one and share invite link here: TODO
- **Code review:** GitHub PRs — require 1 review before merge to `main`. Branch naming: `<member>/<task>` (e.g. `member1/retrieval-build`).
- **Emergency:** phone numbers in a pinned message on the chosen chat.

## GPU server

- **Provider:** TODO (university cluster / Lambda / RunPod / personal rig?)
- **Hostname / address:** TODO
- **SSH command:** TODO (e.g., `ssh <user>@<host> -p <port>`)
- **Who has shell accounts:** TODO (all three members, ideally)
- **Workspace path on server:** `[PROPOSED]` `~/rl-rag/` (clone of this repo)
- **Shared venv path:** `~/rl-rag-venv` (per `scripts/launch_run.sh`)
- **GPU model + VRAM:** TODO (`nvidia-smi` output)
- **CUDA driver version:** TODO (needed to pick compatible vllm wheel)

> ⚠ **Blocking:** no server access → no walking skeleton (Task 1.1) → no Week-2 Gate → nothing after Week 2 can start.

## Groq API key

- **Account owner:** TODO (one member creates the account at <https://groq.com>; the free tier is sufficient for the planned ~1k judge calls)
- **Model used:** `llama-3.3-70b-versatile` (hardcoded in `src/eval/judge.py`)
- **Key shared via:** `[PROPOSED]` 1Password / Bitwarden shared vault — **do not** paste into Discord/Slack/email
- **Verified with curl test (Task 0.1 Step 3):** [ ] Yes / [ ] No
  ```bash
  curl -s https://api.groq.com/openai/v1/chat/completions \
    -H "Authorization: Bearer $GROQ_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":"ping"}]}' \
    | head -c 400
  ```
- **Env var name:** `GROQ_API_KEY` (read in `src/eval/judge.py:28`)

## Experiment tracking

Pick one:

- [ ] `[PROPOSED]` W&B team account — free for academic/student teams. Share URL after creation: TODO
- [ ] Local tensorboard only (results synced via mechanism below)

> Rationale for W&B: the plan runs 4–6 training variants in a grid (Week 4) + ablations (Week 6). W&B's run comparison UI is worth the 10-minute setup.

## Results sync mechanism

Pick one (recommend NFS or rsync; `git-lfs` adds friction):

- [ ] `[PROPOSED]` `rsync` to a shared cloud folder — command: TODO, cadence: nightly
  ```bash
  rsync -av --exclude='*.ckpt' --exclude='*.safetensors' results/ <remote>:<path>/results/
  ```
- [ ] Shared NFS mount — path: TODO
- [ ] W&B artifacts (if W&B chosen above)
- [ ] `git-lfs` — **not recommended** (checkpoints too large; GitHub LFS has a 1 GB/month free quota)

## Weekly sync meeting

- **Day / time:** `[PROPOSED]` Sundays, 6pm Eastern (end of each week; aligns with the 7-week plan's week-boundaries)
- **Where:** TODO (Zoom / Google Meet link)
- **Duration:** 30 min
- **Standing agenda:**
  1. Last week's week-gate — pass/fail/needs-pair-debug (5 min)
  2. Blockers (round-robin, 2 min each)
  3. Next-week ownership + handoffs (10 min)
  4. Any ad-hoc technical topic (5 min)

## Day-1 kickoff checklist (30 min call)

Order matters — item 2 blocks item 3; item 3 blocks item 5.

- [ ] **1.** Agree on Member 1 / 2 / 3 names (5 min)
- [ ] **2.** Confirm GPU server details → fill the GPU server section above (5 min)
- [ ] **3.** All three members SSH to the server + `whoami` + `nvidia-smi` works (5 min)
- [ ] **4.** Pick Groq account owner → owner creates account during the call → shares key via chosen vault (5 min)
- [ ] **5.** Everyone pulls `.env` template or `export GROQ_API_KEY=…` in their shell → curl test returns 200 (3 min)
- [ ] **6.** Pick communication channel (Discord / Signal) + invite everyone (2 min)
- [ ] **7.** Pick W&B vs tensorboard; if W&B, create team space (3 min)
- [ ] **8.** Confirm weekly meeting slot (1 min)
- [ ] **9.** Commit this filled `docs/team.md` back to `main` (1 min)

## Checkpoint — nothing after Week 1 moves until all three are green

- [ ] All three members confirmed SSH access to GPU server
- [ ] Groq curl test returns `200` with a non-empty reply
- [ ] Results-sync mechanism tested with a dummy file (push a text file both ways)

## Changelog

- 2026-10-11: initial skeleton created (Day 1 scaffold session).
- 2026-10-20: expanded with proposed defaults, repo URL, day-1 checklist, key links. Draft ready for kickoff review.
