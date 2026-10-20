# Project Status

**As of:** 2026-10-20
**Branch:** `main`
**Commits:** 4 (initial scaffold → HANDOFF follow-ups → R1/R2/audit → Week-4 code)
**Test state:** `88 passed, 1 deselected` (fast suite; slow dense-retriever test needs server-side stack alignment)
**Local env:** Python 3.12.1, torch 2.5.1+cu121, CUDA available on 1× RTX 4050 Laptop (6 GB VRAM)

---

## ✅ What's been done

### Weeks 0-2 scaffolding (inherited, verified)
All pre-existing scaffold tasks confirmed green on local env:

| Task | Deliverable | Status |
|---|---|---|
| 0.2 | Repo skeleton (`src/`, `tests/`, `configs/`, `docs/`, `data/`, `results/`, `scripts/`) | ✅ |
| 1.2 | `src/common/types.py`, `src/common/parse.py`, `src/data/hotpot.py` | ✅ |
| 1.3 | `src/prompts/templates.py`, `src/rewards/r0_grace.py`, `src/rewards/metrics.py` | ✅ |
| 1.4 | `src/training/run_naming.py`, `scripts/launch_run.sh`, `docs/results-convention.md` | ✅ |
| 2.1 | `src/data/retrievers.py`, `src/data/balanced_split.py`, `src/data/cache.py` (**code only**) | ✅ Code / ⏳ server run |
| 2.2 | R0 edge-case tests, `docs/rewards.md` | ✅ |
| 2.3 | `src/eval/metrics.py`, `src/eval/evaluate.py`, `configs/r0_grpo.yaml` (**code only**) | ✅ Code / ⏳ server run |

### This session's additions

| Task | Deliverable | Commit |
|---|---|---|
| 0.0 | `git init`, `.claude/.idea/.vscode/` added to `.gitignore` | `4c6cca5` |
| 0.1 (partial) | `docs/team.md` draft skeleton — TODOs for humans to fill | `4c6cca5` |
| 1.1 unblock | `scripts/setup_env.sh` patched with `VERL_COMMIT=4aa02fe1663d8048b9c204345b2abe5197870df3` (GRACE pinned) | `4c6cca5` |
| HANDOFF cleanup | 4 new `_hotpot_row_to_sample` tests + 3 `build_prompt↔parse_response` round-trip tests | `e060ef4` |
| Dep safety | `transformers>=4.44,<5` pinned (torch 2.5 can't call `torch.accelerator`) | `e060ef4` |
| **3.2 (full)** | `src/rewards/r1_cost_sensitive.py` + `r2_hard_gated.py` + `audit.py` + 14 tests | `bd238a1` |
| **4.2 (full)** | `src/eval/aggregate.py` + `scripts/make_comparison_table.py` + 3 tests | `9c875d5` |
| **4.3 code side** | `scripts/run_audits.py` (needs real run outputs to produce JSON) | `9c875d5` |
| **4.4 stretch** | `src/data/qasper.py` + 10 tests | `9c875d5` |

### Local pipeline proven
`scripts/local_sanity.py --n 3` ran end-to-end on the laptop GPU:
- Qwen3-0.6B downloaded + loaded in fp16
- 37s/sample inference
- Data → prompt → generate → parse → score chain confirmed healthy
- Output at `results/local_sanity/20261018-162826/`

---

## 🟡 What's remaining — ordered by criticality

### Blocked on humans (30 min call)

- **Task 0.1 completion** — fill the TODOs in `docs/team.md`:
  Who is Member 1 / 2 / 3; who creates the Groq account and shares the key; GPU server hostname + SSH access list; results sync mechanism (recommend rsync); weekly meeting slot. **Nothing after this moves without the decisions above.**

- **Push repo to a remote** so teammates can clone. Create empty repo on GitHub/GitLab, then `git remote add origin <url> && git push -u origin main`.

### Blocked on GPU server

- **Task 1.1 — walking skeleton** *(all 3 members, 1-3 days)*
  `bash scripts/setup_env.sh` → `bash scripts/launch_run.sh configs/skeleton.yaml` → iterate on verl YAML field names until a single step completes. Record working config in `docs/log/<date>-skeleton-run.md`.

- **Task 2.1 server portion** *(Member 1, ~1 day)*
  `python scripts/download_hotpot_full.py` → `python scripts/build_retrieval.py` → verify `0.1 < ans_rate < 0.95` assertion holds for all 4 (bm25|dense) × (train|val) files.

- **Task 2.3 server portion — Week 2 Gate** *(Member 3, 2-6h GPU)*
  First full `r0_grpo` run → evaluate. **Balanced Accuracy must land in [0.71, 0.91].** If not, team halts and pair-debugs before Week 3.

### Blocked on Groq API key

- **Task 3.1 — Groq LJ** *(Member 1)*
  `src/eval/judge.py` (API call + 1 week on-disk cache) + 50-sample hand-labeled validation set + `scripts/validate_lj.py` (≥85% agreement gate) + extend `evaluate.py` to include BERTScore + LJ. Can't be built or validated without a real key.

### Blocked on verl field-name reality

- **Task 3.3 — DAPO configs + grid runner** *(Member 3)*
  Configs `r0_dapo.yaml`, `r1_grpo.yaml`, `r1_dapo.yaml`, `r2_grpo.yaml`, `r2_dapo.yaml`, `grid.yaml`; `scripts/run_grid.sh`. The plan explicitly says to iterate against actual verl errors — writing field names before the walking skeleton proves what verl accepts creates rework.

### Blocked on completed runs

- **Task 4.1 — must-have grid execution** *(Member 3, ~1-2 days wall time)*
- **Task 4.3 Steps 3-6 — audit analysis + `docs/audit-week4.md` + report section** *(Member 2)*
- **Task 5.1 — R1+DAPO, R2+DAPO, Qwen3-1.7B hero run** *(Member 3)*
- **Task 5.2 — QASPER cross-domain run + figures** *(Member 1)*
- **Task 5.3 — final audits + failure cases** *(Member 2)*
- **Task 5.4 — report assembly** *(whole team)*
- **Week 6 — buffer / ablations**
- **Week 7 — final polish**

---

## 🔧 What else IS possible on this laptop

Lower-priority items that don't advance the critical path but are still runnable here:

1. **More local-sanity samples.** Scale `scripts/local_sanity.py --n 10` or `--n 20` on the 6GB GPU; ~5-10 min. Confirms pipeline survives longer batches and gives you a pre-RL baseline for `balanced_accuracy`, `path_shares`, etc.

2. **Full HotpotQA retrieval build.** Likely possible — BM25 is pure CPU; `Qwen3-Embedding-0.6B` dense retriever is ~1.2GB VRAM. 2000+500 queries × ~0.1s on GPU ≈ 4-6 min. Would produce the 4 `data/cache/*.jsonl` files that unblock the server training step. **Caveat:** the slow dense-retriever test fails due to the stack misalignment I had to work around; proper fix (torch 2.6 upgrade or whole-stack downgrade) needs to happen first.

3. **BERTScore baseline.** It's already a dep. Can be computed on the 50-sample tiny set now to validate the eval harness's new metrics plumbing before Task 3.1 wires in Groq LJ.

4. **Download Qwen3-1.7B.** The hero model is only ~3.5 GB. Pulling it now while idle shaves time off Week 5. (**Not** running inference on it — 1.7B in fp16 needs ~3.5 GB weights + activations; may OOM on a 6 GB laptop GPU running other things.)

5. **Draft the Week 5 report scaffolding.** Can create `docs/report/REPORT.md` outline + section stubs (`reward-design.md`, `training.md`, `results.md`, `failure-cases.md`) with placeholder tables matching the plan.

6. **Draft the Groq judge skeleton without validation.** `src/eval/judge.py` structure (caching + prompt + response parsing) can be written without a key; just can't validate. Would save Member 1 a few hours in Week 3.

7. **Delete the empty `requirements.txt`** or populate it from pyproject. Trivial housekeeping; left it alone because it's committed.

8. **`docs/log/` templates.** Pre-create `docs/log/<date>-skeleton-run.md` and `docs/log/<date>-week2-gate.md` templates so members just fill in evidence rather than inventing structure under pressure.

9. **Alignment commits for the HF dep stack.** Either upgrade torch to `>=2.6` (unlocks `sentence-transformers 6.x`, `datasets 5.x`, `transformers 5.x` all together — modern unified stack), OR downgrade everything (`sentence-transformers<6`, `datasets<5`, keep `transformers<5`). Needs a team decision because it propagates to server install.

---

## ❌ What is NOT possible on this laptop

| Task | Why not |
|---|---|
| **Install `verl`** | Uses Ray + Linux-oriented tooling; `scripts/setup_env.sh` is bash + assumes `python3.12` and CUDA 12.4 wheels — Windows CUDA 12.1 wheels work for torch but not for verl's build chain |
| **Install `vllm` for training** | Historically Linux-only; experimental Windows path is unstable and would deviate from GRACE's reference setup |
| **GRPO or DAPO training run** | Needs rollouts + policy grads + optimizer state in VRAM simultaneously. Even Qwen3-0.6B with LoRA + grad checkpointing typically needs ~8-12 GB VRAM; this laptop has 6 GB |
| **Qwen3-1.7B hero run** | ~3.5 GB weights in fp16, then +grads for training; firmly past 6 GB |
| **Validate verl config field names** | Can't test `configs/skeleton.yaml` or `configs/r0_grpo.yaml` without a working verl install |
| **Resolve the Week 2 Gate** | By definition requires completing a full `r0_grpo` run and evaluating |
| **Run the `must`/`should` grids** | 4-6 runs × 2-6h each = GPU server territory |
| **Hand-label the 50-example LJ validation set** | Human judgment task (Member 1) |
| **Agree on team logistics** | Human-only |
| **Sign up for Groq account + verify key** | Needs the designated member to go to groq.com |
| **SSH into the GPU server** | Hostname + access list are TODOs in `docs/team.md` |

---

## 🎯 Recommended next moves (in order)

1. **You personally, today:** fill in `docs/team.md`, push repo to GitHub/GitLab, send the clone URL + team.md to your two teammates. (30 min)
2. **Team, this week:** meet for 30 min, agree Task 0.1 decisions, verify SSH + Groq key.
3. **Team, on server:** `scripts/setup_env.sh` → walking skeleton debug session.
4. **Member 1, on server:** full retrieval build.
5. **Member 3, on server:** first R0+GRPO run; check Week 2 Gate.
6. **Optional parallel work on this laptop while that happens:** items 2 (full retrieval build local), 4 (download 1.7B), 5 (report scaffolding), 8 (log templates) from the "possible" list above.
