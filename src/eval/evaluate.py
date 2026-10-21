"""CLI: evaluate a checkpoint (or base model) on a split.

Usage:
    # Server (vLLM path, fast):
    python -m src.eval.evaluate --checkpoint path/to/ckpt \
        --split data/cache/dense_val.jsonl \
        --out results/<run_id>/metrics.json

    # Laptop (HF transformers path, works on CPU or small GPU):
    python -m src.eval.evaluate --checkpoint Qwen/Qwen3-0.6B \
        --split data/raw/hotpot_tiny.jsonl \
        --out results/baseline_0.6b/metrics.json \
        --backend hf --device auto --max-samples 20 --max-new-tokens 256

Backends:
    vllm  -- fastest; needs vLLM + CUDA. Default on server.
    hf    -- plain `transformers` + `torch`. Works on CPU or GPU.
    auto  -- try vllm first, fall back to hf.

The module itself is import-safe on a laptop; all heavy deps are lazy.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from statistics import mean
from typing import Literal

from src.common.device import resolve_device, summary as device_summary
from src.common.parse import parse_response
from src.common.types import Sample
from src.eval.metrics import abstention_accuracy, balanced_accuracy, exact_match
from src.prompts.templates import build_prompt

log = logging.getLogger(__name__)

Backend = Literal["vllm", "hf", "auto"]


def _load_split(path: Path, limit: int | None = None) -> list[Sample]:
    out: list[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            row.pop("prompt", None)  # verl-only column; drop before Sample(**row)
            out.append(Sample(**row))
    return out


def _try_vllm(
    checkpoint: str, prompts: list[str], max_new_tokens: int
) -> list[str] | None:
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except ImportError:
        return None
    llm = LLM(model=checkpoint, dtype="float16", gpu_memory_utilization=0.6)
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outs = llm.generate(prompts, params)
    return [o.outputs[0].text for o in outs]


def _hf_generate(
    checkpoint: str,
    prompts: list[str],
    *,
    device: str,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    """Local HF path. Imports LocalRunner lazily so the module-level import
    cost stays low."""
    from src.eval.local_runner import GenConfig, LocalRunner

    runner = LocalRunner(checkpoint, device=device)
    cfg = GenConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        batch_size=batch_size,
    )

    def _tick(done: int, total: int) -> None:
        sys.stderr.write(f"\r  gen {done}/{total}")
        sys.stderr.flush()

    out = runner.generate(prompts, cfg, progress=_tick)
    sys.stderr.write("\n")
    return out


def generate(
    checkpoint: str,
    prompts: list[str],
    *,
    backend: Backend = "auto",
    device: str = "auto",
    max_new_tokens: int = 512,
    batch_size: int = 1,
) -> list[str]:
    """Backend dispatcher. See module docstring for backend semantics."""
    if backend == "vllm":
        got = _try_vllm(checkpoint, prompts, max_new_tokens)
        if got is None:
            raise RuntimeError(
                "backend='vllm' requested but vllm is not importable."
            )
        return got
    if backend == "hf":
        return _hf_generate(
            checkpoint, prompts,
            device=device, max_new_tokens=max_new_tokens, batch_size=batch_size,
        )
    # auto
    got = _try_vllm(checkpoint, prompts, max_new_tokens)
    if got is not None:
        return got
    log.info("vLLM unavailable; falling back to HF transformers backend.")
    return _hf_generate(
        checkpoint, prompts,
        device=device, max_new_tokens=max_new_tokens, batch_size=batch_size,
    )


def _bertscore_f1(preds: list[str], refs: list[str]) -> list[float]:
    """Compute BERTScore F1 per (pred, ref). Lazy-imports bert_score so the
    module stays import-safe on machines without it installed."""
    from bert_score import score as bs_score  # type: ignore
    _, _, f1 = bs_score(preds, refs, lang="en", rescale_with_baseline=False, verbose=False)
    return [float(x) for x in f1.tolist()]


def _judge_scores(
    questions: list[str],
    preds: list[str],
    refs: list[str],
    cache_dir: Path | None,
) -> list[float]:
    """Call GroqJudge on each (question, pred, ref). Reads GROQ_API_KEY from env."""
    from src.eval.judge import GroqJudge
    judge = GroqJudge(cache_dir=cache_dir)
    return [judge.score(q, p, r) for q, p, r in zip(questions, preds, refs)]


def _score(
    samples: list[Sample],
    responses: list[str],
    *,
    with_bertscore: bool = False,
    with_judge: bool = False,
    judge_cache_dir: Path | None = None,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for s, resp in zip(samples, responses):
        parsed = parse_response(resp)
        em = exact_match(parsed.answer, s.gold_answer) if s.is_answerable else 0.0
        rows.append({
            "sample_id": s.sample_id,
            "is_answerable": s.is_answerable,
            "path": parsed.path,
            "quoted_evidence": parsed.quoted_evidence,
            "answer": parsed.answer,
            "raw_response": resp,
            "em": em,
        })

    if with_bertscore or with_judge:
        ans_indices = [i for i, r in enumerate(rows) if r["is_answerable"]]
        if ans_indices:
            preds = [rows[i]["answer"] for i in ans_indices]
            refs = [samples[i].gold_answer for i in ans_indices]
            questions = [samples[i].question for i in ans_indices]
            if with_bertscore:
                f1s = _bertscore_f1(preds, refs)
                for i, f1 in zip(ans_indices, f1s):
                    rows[i]["bertscore_f1"] = f1
            if with_judge:
                judge_vals = _judge_scores(questions, preds, refs, judge_cache_dir)
                for i, j in zip(ans_indices, judge_vals):
                    rows[i]["judge_score"] = j
        # Unanswerable rows get None for both — don't fabricate a score.
        for r in rows:
            if not r["is_answerable"]:
                if with_bertscore:
                    r["bertscore_f1"] = None
                if with_judge:
                    r["judge_score"] = None

    ans_rows = [r for r in rows if r["is_answerable"]]
    em_ans = mean(r["em"] for r in ans_rows) if ans_rows else 0.0
    abst = abstention_accuracy([(r["path"], r["is_answerable"]) for r in rows])
    bal = balanced_accuracy(em_ans, abst)
    n = len(rows)
    metrics: dict = {
        "n": n,
        "n_answerable": len(ans_rows),
        "em_answerable": em_ans,
        "abstention_unanswerable": abst,
        "balanced_accuracy": bal,
        "path_shares": {
            "evidence": sum(1 for r in rows if r["path"] == "evidence") / n if n else 0.0,
            "llm":      sum(1 for r in rows if r["path"] == "llm") / n if n else 0.0,
            "invalid":  sum(1 for r in rows if r["path"] == "invalid") / n if n else 0.0,
        },
    }
    if with_bertscore and ans_rows:
        vals = [r["bertscore_f1"] for r in ans_rows if r.get("bertscore_f1") is not None]
        metrics["bertscore_f1_answerable"] = mean(vals) if vals else 0.0
    if with_judge and ans_rows:
        vals = [r["judge_score"] for r in ans_rows if r.get("judge_score") is not None]
        metrics["judge_score_answerable"] = mean(vals) if vals else 0.0
    return rows, metrics


def evaluate(
    checkpoint: str,
    split_file: Path,
    out: Path,
    *,
    backend: Backend = "auto",
    device: str = "auto",
    max_samples: int | None = None,
    max_new_tokens: int = 512,
    batch_size: int = 1,
    with_bertscore: bool = False,
    with_judge: bool = False,
    judge_cache_dir: Path | None = None,
) -> dict:
    samples = _load_split(split_file, limit=max_samples)
    prompts = [build_prompt(s.question, s.retrieved_passages) for s in samples]
    responses = generate(
        checkpoint, prompts,
        backend=backend, device=device,
        max_new_tokens=max_new_tokens, batch_size=batch_size,
    )
    rows, metrics = _score(
        samples, responses,
        with_bertscore=with_bertscore,
        with_judge=with_judge,
        judge_cache_dir=judge_cache_dir,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out.parent / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    return metrics


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Model id or local path (e.g. Qwen/Qwen3-0.6B).")
    ap.add_argument("--split", required=True, type=Path,
                    help="Path to a .jsonl split emitted by scripts/build_retrieval.py "
                         "OR scripts/download_hotpot_tiny.py raw file.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output path for metrics.json.")
    ap.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap for quick sanity runs.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--bertscore", action="store_true",
                    help="Compute BERTScore F1 on answerable predictions.")
    ap.add_argument("--judge", action="store_true",
                    help="Call the Groq LLM-as-Judge on answerable predictions. "
                         "Requires GROQ_API_KEY in the environment.")
    ap.add_argument("--judge-cache-dir", type=Path, default=Path("data/cache/judge"),
                    help="On-disk cache dir for the LLM-as-Judge responses.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log.info("device: %s", device_summary())
    log.info("resolved: %s", resolve_device(args.device))

    m = evaluate(
        args.checkpoint, args.split, args.out,
        backend=args.backend, device=args.device,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        with_bertscore=args.bertscore,
        with_judge=args.judge,
        judge_cache_dir=args.judge_cache_dir if args.judge else None,
    )
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    _cli()
