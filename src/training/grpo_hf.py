"""Self-contained GRPO trainer: transformers + peft only. No verl, vLLM or ray.

This is the fallback path. It trains the same thing the verl configs train
(Qwen3-0.6B + LoRA, group-relative advantages, our R0/R1/R2 rewards on the
BM25/dense retrieval cache) but with nothing in the loop that can fail to
install: rollouts come from `model.generate`, the update is a PPO-clip step
on the LoRA weights.

    uv run python -m src.training.grpo_hf --reward r0 --steps 200

Checkpoints are plain PEFT adapters in results/<run_id>/checkpoints/global_step_N,
so `python -m src.eval.evaluate --checkpoint <that dir> --backend hf` loads them.

Per step:
  1. sample `batch_size` prompts, generate `n` completions each (sampling)
  2. reward every completion with the chosen reward function
  3. advantage = (r - group mean) / (group std + eps)       <- the "GR" in GRPO
  4. PPO-clip policy loss on completion tokens, + optional KL-to-base penalty
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

# glibc's malloc arenas multiply per thread and can inflate RSS by 3-5x on a
# CPU box with no GPU (observed: 6.2 GB resident for a 0.6B model doing
# nothing special). Must be set before numpy/torch are imported anywhere in
# the process, including by a parent script -- harmless on CUDA, where the
# model and activations live in VRAM instead.
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

from src.common.types import Sample
from src.prompts.templates import build_prompt
from src.rewards.r0_grace import r0_grace
from src.rewards.r1_cost_sensitive import r1_cost_sensitive
from src.rewards.r2_hard_gated import r2_hard_gated
from src.training.run_naming import make_run_id

REWARDS = {"r0": r0_grace, "r1": r1_cost_sensitive, "r2": r2_hard_gated}


def load_rows(path: Path, limit: int | None = None) -> list[tuple[str, Sample]]:
    out: list[tuple[str, Sample]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            prompt = row.pop("prompt", None)
            sample = Sample(**row)
            out.append((prompt or build_prompt(sample.question, sample.retrieved_passages), sample))
    if not out:
        raise SystemExit(f"{path} is empty")
    return out


def group_advantages(rewards: list[float], n: int, eps: float = 1e-6) -> list[float]:
    """Normalise rewards inside each group of `n` rollouts of the same prompt."""
    adv: list[float] = []
    for g in range(0, len(rewards), n):
        grp = rewards[g:g + n]
        mean = sum(grp) / len(grp)
        std = math.sqrt(sum((r - mean) ** 2 for r in grp) / len(grp))
        adv.extend((r - mean) / (std + eps) for r in grp)
    return adv


def find_resume(results_dir: Path, prefix: str) -> tuple[Path, Path, int] | None:
    """Newest run dir with this prefix that has a checkpoint -> (run, ckpt, step)."""
    for run in sorted(results_dir.glob(f"{prefix}_*"), reverse=True):
        ckpts = sorted(
            (p for p in (run / "checkpoints").glob("global_step_*") if (p / "adapter_config.json").exists()),
            key=lambda p: int(p.name.rsplit("_", 1)[-1]),
        )
        if ckpts:
            return run, ckpts[-1], int(ckpts[-1].name.rsplit("_", 1)[-1])
    return None



def _rss_log(tag: str) -> None:
    if os.environ.get("GRPO_HF_MEM_DEBUG"):
        import resource
        print(f"[mem] {tag}: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024:.2f} GB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--model-tag", default="qwen3-0.6b", help="short name used in the run id")
    ap.add_argument("--train", type=Path, default=Path("data/cache/bm25_train.jsonl"))
    ap.add_argument("--reward", choices=sorted(REWARDS), default="r0")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8, help="prompts per step")
    ap.add_argument("--n", type=int, default=4, help="rollouts per prompt")
    ap.add_argument("--max-prompt-tokens", type=int, default=1536)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip-ratio", type=float, default=0.2)
    ap.add_argument("--kl-coef", type=float, default=0.0,
                    help=">0 adds a KL(policy || base) penalty; costs one extra forward pass")
    ap.add_argument("--micro-batch", type=int, default=4, help="sequences per forward/backward")
    ap.add_argument("--gen-batch", type=int, default=16, help="sequences per generate() call")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--save-freq", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--model-dtype", choices=["float32", "bfloat16"], default="float32",
                    help="master-weight dtype; bfloat16 roughly halves RAM/VRAM "
                         "(fine for a smoke test, use float32 for a real run on GPU)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training rows")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.common.device import resolve_device

    device = resolve_device(args.device)
    use_amp = device == "cuda"
    if device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    prefix = f"{args.reward}_hfgrpo_{args.model_tag}_s{args.seed}"
    resume = None if args.no_resume else find_resume(args.results_dir, prefix)
    if resume:
        run_dir, resume_ckpt, start_step = resume
        print(f"Resuming {run_dir.name} from step {start_step}")
    else:
        run_dir = args.results_dir / make_run_id(args.reward, "hfgrpo", args.model_tag, args.seed)
        resume_ckpt, start_step = None, 0
    ckpt_root = run_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    log_f = (run_dir / "train_metrics.jsonl").open("a", encoding="utf-8")

    _rss_log("before tokenizer")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # required for batched generation
    tok.truncation_side = "left"

    # fp32 weights by default; on GPU the forward passes run under fp16 autocast
    # regardless. --model-dtype bfloat16 halves RAM/VRAM for constrained boxes.
    master_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.model_dtype]
    _rss_log("before model load")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=master_dtype, low_cpu_mem_usage=True
    )
    if resume_ckpt is not None:
        model = PeftModel.from_pretrained(base, str(resume_ckpt), is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            target_modules="all-linear", task_type="CAUSAL_LM",
        ))
    _rss_log("after model load")
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    if resume_ckpt is not None and (resume_ckpt / "optimizer.pt").exists():
        opt.load_state_dict(torch.load(resume_ckpt / "optimizer.pt", map_location=device))
    scaler = torch.amp.GradScaler(enabled=use_amp)
    print(f"device={device}  trainable params={sum(p.numel() for p in params):,}  run_dir={run_dir}")

    rows = load_rows(args.train, args.limit)
    order_rng = random.Random(args.seed)
    reward_fn = REWARDS[args.reward]

    def completion_logprobs(seqs, attn, comp_mask):
        """Per-token log-prob of each token given its prefix. [B, T-1]."""
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=seqs, attention_mask=attn).logits[:, :-1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        return logp.gather(-1, seqs[:, 1:].unsqueeze(-1)).squeeze(-1) * comp_mask[:, 1:]

    def save(step: int) -> None:
        d = ckpt_root / f"global_step_{step}"
        model.save_pretrained(str(d))
        tok.save_pretrained(str(d))
        torch.save(opt.state_dict(), d / "optimizer.pt")
        print(f"  saved {d}")

    _rss_log("before training loop")
    for step in range(start_step + 1, args.steps + 1):
        t0 = time.time()
        batch = [rows[order_rng.randrange(len(rows))] for _ in range(args.batch_size)]
        prompts = [p for p, _ in batch for _ in range(args.n)]
        samples = [s for _, s in batch for _ in range(args.n)]

        # ---- 1. rollouts -------------------------------------------------
        _rss_log(f"step {step} before generate")
        model.eval()
        texts: list[str] = []
        items: list[tuple] = []   # (prompt_ids, completion_ids) per sequence
        with torch.no_grad():
            for i in range(0, len(prompts), args.gen_batch):
                enc = tok(prompts[i:i + args.gen_batch], return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_prompt_tokens).to(device)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    gen = model.generate(
                        **enc, do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                        max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
                    )
                plen = enc["input_ids"].shape[1]
                for b in range(gen.shape[0]):
                    comp = gen[b, plen:]
                    # cut at first EOS (keep it: the policy is trained to stop)
                    eos = (comp == tok.eos_token_id).nonzero()
                    comp = comp[: int(eos[0]) + 1] if len(eos) else comp
                    p_ids = enc["input_ids"][b][enc["attention_mask"][b].bool()]
                    items.append((p_ids.cpu(), comp.cpu()))
                    texts.append(tok.decode(comp, skip_special_tokens=True))

        # ---- 2. rewards + 3. group advantages ----------------------------
        _rss_log(f"step {step} after generate")
        scored = [reward_fn(t, s) for t, s in zip(texts, samples)]
        rewards = [float(r) for r, _ in scored]
        advs = group_advantages(rewards, args.n)

        # ---- 4. PPO-clip update -----------------------------------------
        def collate(idx: list[int]):
            L = max(len(items[j][0]) + len(items[j][1]) for j in idx)
            seqs = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(idx), L), dtype=torch.long)
            cmask = torch.zeros((len(idx), L), dtype=torch.float32)
            for r, j in enumerate(idx):
                p, c = items[j]
                n_tok = len(p) + len(c)
                seqs[r, :n_tok] = torch.cat([p, c])     # right-pad for training
                attn[r, :n_tok] = 1
                cmask[r, len(p):n_tok] = 1.0
            return seqs.to(device), attn.to(device), cmask.to(device)

        micro = [list(range(i, min(i + args.micro_batch, len(items))))
                 for i in range(0, len(items), args.micro_batch)]
        micro = [m for m in micro if any(len(items[j][1]) > 0 for j in m)]

        # log-probs under the rollout policy (no dropout, so eval == train here)
        old_lp = []
        with torch.no_grad():
            for idx in micro:
                seqs, attn, cmask = collate(idx)
                old_lp.append(completion_logprobs(seqs, attn, cmask).cpu())

        _rss_log(f"step {step} before backward")
        model.train()
        opt.zero_grad(set_to_none=True)
        tot_loss = tot_kl = 0.0
        n_tok_total = sum(len(items[j][1]) for m in micro for j in m) or 1
        for idx, olp in zip(micro, old_lp):
            seqs, attn, cmask = collate(idx)
            new_lp = completion_logprobs(seqs, attn, cmask)
            m = cmask[:, 1:]
            adv = torch.tensor([advs[j] for j in idx], device=device).unsqueeze(1)
            ratio = torch.exp(new_lp - olp.to(device))
            surr = torch.minimum(ratio * adv,
                                 torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio) * adv)
            loss = -(surr * m).sum() / n_tok_total          # token-mean over the whole batch
            if args.kl_coef > 0:
                with torch.no_grad(), model.disable_adapter():
                    ref_lp = completion_logprobs(seqs, attn, cmask)
                d = ref_lp - new_lp                           # low-variance KL (k3)
                kl = ((torch.exp(d) - d - 1).clamp(max=10) * m).sum() / n_tok_total
                loss = loss + args.kl_coef * kl
                tot_kl += float(kl)
            scaler.scale(loss).backward()
            tot_loss += float(loss)
        scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        scaler.step(opt)
        scaler.update()

        paths = [p["path"] for _, p in scored]
        rec = {
            "step": step,
            "reward_mean": sum(rewards) / len(rewards),
            "reward_max": max(rewards),
            "format_valid": sum(p["format_valid"] for _, p in scored) / len(scored),
            "path_evidence": paths.count("evidence") / len(paths),
            "path_llm": paths.count("llm") / len(paths),
            "loss": tot_loss,
            "kl": tot_kl,
            "grad_norm": grad_norm,
            "resp_tokens_mean": sum(len(c) for _, c in items) / len(items),
            "sec": round(time.time() - t0, 1),
        }
        _rss_log(f"step {step} after update")
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        if step == start_step + 1:
            print("  sample response:", repr(texts[0][:200]))

        if step % args.save_freq == 0 or step == args.steps:
            save(step)

    print(f"done. run_dir={run_dir}")


if __name__ == "__main__":
    main()
