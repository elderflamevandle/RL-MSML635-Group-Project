"""R1 — Cost-sensitive reward with hallucination penalty + quote-containment gate."""
from dataclasses import dataclass
from src.common.types import Sample
from src.common.parse import parse_response
from src.rewards.metrics import rouge_l, quote_in_passages


@dataclass(frozen=True)
class R1Config:
    Rf: float = 0.5
    Rp: float = 1.0
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.5
    lam: float = 1.0


def r1_cost_sensitive(
    response: str, sample: Sample, cfg: R1Config = R1Config()
) -> tuple[float, dict]:
    parsed = parse_response(response)
    parts: dict = {
        "path": parsed.path,
        "format_valid": parsed.format_valid,
        "correct_path": False,
        "hallucination": False,
        "quote_grounded": False,
        "rouge_l_evidence": 0.0,
        "rouge_l_answer": 0.0,
    }

    # Hallucination = answered (evidence path) on unanswerable sample.
    if parsed.path == "evidence" and not sample.is_answerable:
        parts["hallucination"] = True
        return -cfg.lam, parts

    if not parsed.format_valid:
        return 0.0, parts

    correct_path = (
        (parsed.path == "evidence" and sample.is_answerable)
        or (parsed.path == "llm" and not sample.is_answerable)
    )
    parts["correct_path"] = correct_path

    if not correct_path:
        return cfg.Rf, parts

    if parsed.path == "llm":
        rl_ans = rouge_l(parsed.answer, sample.gold_answer)
        parts["rouge_l_answer"] = rl_ans
        return cfg.Rf + cfg.Rp + cfg.gamma * rl_ans, parts

    # Correct <evidence> path with quote-containment gate on Rf.
    grounded = all(quote_in_passages(q, sample.retrieved_passages) for q in parsed.quoted_evidence)
    parts["quote_grounded"] = grounded
    rl_ev = rouge_l(" ".join(parsed.quoted_evidence), sample.gold_evidence)
    rl_ans = rouge_l(parsed.answer, sample.gold_answer)
    parts["rouge_l_evidence"] = rl_ev
    parts["rouge_l_answer"] = rl_ans
    rf_gated = cfg.Rf if grounded else 0.0
    return rf_gated + cfg.Rp + cfg.alpha * rl_ev + cfg.beta * rl_ans, parts
