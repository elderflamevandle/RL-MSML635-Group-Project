"""R2 — Hard-gated multiplicative reward. The RQ1 extreme test."""
from dataclasses import dataclass
from src.common.types import Sample
from src.common.parse import parse_response
from src.rewards.metrics import rouge_l, quote_in_passages


@dataclass(frozen=True)
class R2Config:
    w_p: float = 0.4
    w_q: float = 0.6


def r2_hard_gated(
    response: str, sample: Sample, cfg: R2Config = R2Config()
) -> tuple[float, dict]:
    parsed = parse_response(response)

    # R_format gate: valid tags AND (if evidence path) every quote in retrieved.
    r_format = 1
    if not parsed.format_valid:
        r_format = 0
    elif parsed.path == "evidence":
        if not all(quote_in_passages(q, sample.retrieved_passages) for q in parsed.quoted_evidence):
            r_format = 0

    # R_path gate: correct choice given is_answerable.
    correct_path = (
        (parsed.path == "evidence" and sample.is_answerable)
        or (parsed.path == "llm" and not sample.is_answerable)
    )
    r_path = 1 if correct_path else 0

    # R_quality in [0,1].
    r_quality = 0.0
    if r_format == 1 and r_path == 1:
        if parsed.path == "evidence":
            rl_ev = rouge_l(" ".join(parsed.quoted_evidence), sample.gold_evidence)
            rl_ans = rouge_l(parsed.answer, sample.gold_answer)
            r_quality = 0.5 * rl_ev + 0.5 * rl_ans
        else:  # llm path
            r_quality = rouge_l(parsed.answer, sample.gold_answer)

    reward = r_format * (cfg.w_p * r_path + cfg.w_q * r_path * r_quality)
    parts = {
        "path": parsed.path,
        "R_format": r_format,
        "R_path": r_path,
        "R_quality": r_quality,
    }
    return reward, parts
