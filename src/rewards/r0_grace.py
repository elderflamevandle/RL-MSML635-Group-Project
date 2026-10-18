"""R0 — GRACE baseline reward.

R0 = 0,                                    invalid format
     Rf,                                   valid format, wrong path
     Rf + Rp,                              correct <llm> path
     Rf + Rp + α·RLev + β·RLans,           correct <evidence> path
"""
from dataclasses import dataclass
from src.common.types import Sample
from src.common.parse import parse_response
from src.rewards.metrics import rouge_l


@dataclass(frozen=True)
class R0Config:
    Rf: float = 0.5
    Rp: float = 1.0
    alpha: float = 0.5
    beta: float = 0.5


def r0_grace(
    response: str, sample: Sample, cfg: R0Config | None = None
) -> tuple[float, dict]:
    if cfg is None:
        cfg = R0Config()
    parsed = parse_response(response)
    parts: dict = {
        "path": parsed.path,
        "format_valid": parsed.format_valid,
        "correct_path": None,          # None = undefined (invalid format); True/False only after format_valid is confirmed.
        "rouge_l_evidence": 0.0,
        "rouge_l_answer": 0.0,
    }

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
        return cfg.Rf + cfg.Rp, parts

    # Correct <evidence> path.
    rl_ev = rouge_l(" ".join(parsed.quoted_evidence), sample.gold_evidence)
    rl_ans = rouge_l(parsed.answer, sample.gold_answer)
    parts["rouge_l_evidence"] = rl_ev
    parts["rouge_l_answer"] = rl_ans
    return cfg.Rf + cfg.Rp + cfg.alpha * rl_ev + cfg.beta * rl_ans, parts
