"""Local sanity check: run the FULL pipeline against the base model.

Why this exists:
    Before anyone touches the GPU server, we want to prove the data +
    prompt + parse + eval pipeline is wired correctly. We do that by
    running the BASE (un-trained) Qwen3-0.6B model against a few HotpotQA
    samples locally and checking that:

    (a) the model produces something,
    (b) `parse_response` recognizes the format most of the time,
    (c) `exact_match` + `abstention_accuracy` + `balanced_accuracy`
        produce sensible numbers (base model is weak, so expect low
        scores -- but non-crashing).

Usage:
    # Fast smoke test (3 samples, CPU, ~1-3 min):
    python scripts/local_sanity.py --n 3

    # Full tiny set (50 samples, CPU, 10-30 min; GPU: 1-3 min):
    python scripts/local_sanity.py --n 50

    # Explicit device:
    python scripts/local_sanity.py --n 10 --device cuda
    python scripts/local_sanity.py --n 10 --device cpu

Output goes to results/local_sanity/<timestamp>/ with metrics.json +
predictions.jsonl.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

# Make this script runnable as `python scripts/local_sanity.py` (not just
# `-m scripts.local_sanity`) by adding the repo root to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.device import resolve_device, summary as device_summary  # noqa: E402
from src.common.parse import parse_response  # noqa: E402
from src.data.hotpot import load_hotpot_tiny  # noqa: E402
from src.eval.local_runner import GenConfig, LocalRunner  # noqa: E402
from src.eval.metrics import (abstention_accuracy, balanced_accuracy,  # noqa: E402
                              exact_match)
from src.prompts.templates import build_prompt  # noqa: E402

log = logging.getLogger("local_sanity")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B",
                    help="HF model id or local path.")
    ap.add_argument("--tiny-file", type=Path,
                    default=Path("data/raw/hotpot_tiny.jsonl"),
                    help="HotpotQA tiny file (output of download_hotpot_tiny.py).")
    ap.add_argument("--n", type=int, default=10,
                    help="How many samples to run (<= 50 for the tiny file).")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="Keep small on CPU to stay fast.")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out-root", type=Path, default=Path("results/local_sanity"))
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # ---- 1. show environment ---------------------------------------
    device = resolve_device(args.device)
    log.info("Device probe: %s", device_summary())
    log.info("Resolved device: %s", device)
    if device == "cpu":
        log.warning(
            "Running on CPU. Expect ~10-30s per sample with max_new_tokens=256."
        )

    # ---- 2. load samples -------------------------------------------
    if not args.tiny_file.exists():
        log.error(
            "%s not found. Run `python scripts/download_hotpot_tiny.py` first.",
            args.tiny_file,
        )
        sys.exit(2)

    samples = load_hotpot_tiny(args.tiny_file)[: args.n]
    log.info("Loaded %d samples from %s", len(samples), args.tiny_file)
    # The tiny file is raw HotpotQA so every sample is answerable.
    n_ans = sum(1 for s in samples if s.is_answerable)
    log.info("Answerable: %d / %d  (sanity set is fully-answerable by design)",
             n_ans, len(samples))

    prompts = [build_prompt(s.question, s.retrieved_passages) for s in samples]

    # ---- 3. generate -----------------------------------------------
    runner = LocalRunner(args.model, device=args.device)
    cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        batch_size=args.batch_size,
    )

    def _tick(done: int, total: int) -> None:
        sys.stderr.write(f"\r  gen {done}/{total}")
        sys.stderr.flush()

    t0 = time.time()
    responses = runner.generate(prompts, cfg, progress=_tick)
    sys.stderr.write("\n")
    elapsed = time.time() - t0
    log.info("Generated %d responses in %.1fs (%.1fs/sample)",
             len(responses), elapsed, elapsed / max(1, len(responses)))

    # ---- 4. score + print each row --------------------------------
    rows = []
    for s, resp in zip(samples, responses):
        parsed = parse_response(resp)
        em = exact_match(parsed.answer, s.gold_answer) if s.is_answerable else 0.0
        rows.append({
            "sample_id": s.sample_id,
            "question": s.question,
            "gold_answer": s.gold_answer,
            "is_answerable": s.is_answerable,
            "path": parsed.path,
            "format_valid": parsed.format_valid,
            "quoted_evidence": parsed.quoted_evidence,
            "answer": parsed.answer,
            "em": em,
            "raw_response": resp,
        })

    # Pretty per-row log
    for r in rows:
        tag = "✓" if r["em"] == 1.0 else ("•" if r["format_valid"] else "✗")
        log.info(
            "%s  [%s]  Q: %s", tag, r["path"], r["question"][:70] + ("…" if len(r["question"]) > 70 else "")
        )
        log.info("       gold:    %s", r["gold_answer"][:80])
        log.info("       pred:    %s", r["answer"][:80])

    # ---- 5. aggregate ---------------------------------------------
    ans_rows = [r for r in rows if r["is_answerable"]]
    em_ans = mean(r["em"] for r in ans_rows) if ans_rows else 0.0
    abst = abstention_accuracy([(r["path"], r["is_answerable"]) for r in rows])
    bal = balanced_accuracy(em_ans, abst)
    n = len(rows)
    metrics = {
        "model": args.model,
        "device": device,
        "n": n,
        "n_answerable": len(ans_rows),
        "em_answerable": em_ans,
        "abstention_unanswerable": abst,
        "balanced_accuracy": bal,
        "elapsed_s": elapsed,
        "path_shares": {
            "evidence": sum(1 for r in rows if r["path"] == "evidence") / n if n else 0.0,
            "llm":      sum(1 for r in rows if r["path"] == "llm") / n if n else 0.0,
            "invalid":  sum(1 for r in rows if r["path"] == "invalid") / n if n else 0.0,
        },
        "format_valid_rate":
            sum(1 for r in rows if r["format_valid"]) / n if n else 0.0,
    }

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out_root / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (out / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )

    log.info("")
    log.info("=== SANITY SUMMARY ===")
    log.info("model: %s on %s", args.model, device)
    log.info("n: %d  (answerable: %d)", n, len(ans_rows))
    log.info("format_valid_rate: %.2f", metrics["format_valid_rate"])
    log.info("em_answerable:     %.3f", em_ans)
    log.info("balanced_accuracy: %.3f", bal)
    log.info("path shares:       %s", metrics["path_shares"])
    log.info("out: %s", out)
    log.info("")
    log.info("Interpretation:")
    if metrics["format_valid_rate"] < 0.1:
        log.info("  Low format_valid_rate — base model isn't following the")
        log.info("  tag schema well. That's expected and is literally what RL")
        log.info("  training will fix. The pipeline is still working.")
    if em_ans < 0.2:
        log.info("  Low em_answerable — base model answers aren't matching gold")
        log.info("  exactly. Also expected without fine-tuning.")
    log.info("  Pipeline health is confirmed as long as this script exited 0.")


if __name__ == "__main__":
    main()
