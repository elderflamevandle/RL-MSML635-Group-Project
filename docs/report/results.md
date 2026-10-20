# Results

**Section owner:** Member 1
**Depends on:** Task 4.2 (aggregate table; code done) + Task 5.2 (QASPER; pending server).

## Main table

Auto-generated: `python scripts/make_comparison_table.py` writes `docs/results-table.md` + `.csv`.

<Paste `docs/results-table.md` here or link to it.>

## Per-dataset breakdown

### HotpotQA (val)

| Reward | Optimizer | BalAcc | EM | TNR | BERT-F1 | LJ |
|---|---|---|---|---|---|---|
| | | | | | | |

### QASPER (val)

| Reward | Optimizer | BalAcc | EM | TNR | BERT-F1 | LJ |
|---|---|---|---|---|---|---|
| | | | | | | |

## Figures

- `docs/figures/path_shares_by_reward.png` — stacked bar: evidence/llm/invalid per reward
- `docs/figures/balacc_vs_optimizer.png` — grouped bar: R0/R1/R2 across GRPO/DAPO
- `docs/figures/hero_scaling.png` — line/bar: 0.6B vs 1.7B hero

## Observations

### RQ1 — reward comparison

<Report winner on `balanced_accuracy` at fixed optimizer. Report whether abstention improved on R1/R2 vs R0.>

### RQ2 — optimizer comparison

<Report winner on sample efficiency.>

### Cross-domain

<Did the HotpotQA winner's `balanced_accuracy` drop under 10 points on QASPER? If yes, policy generalized reasonably. If no, note.>
