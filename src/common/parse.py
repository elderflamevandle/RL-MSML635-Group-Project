"""Response parser. SINGLE SOURCE OF TRUTH for interpreting model outputs.
Both reward functions and the eval harness call this — fix bugs here, not downstream."""
import re
from src.common.types import ParsedResponse

_EVIDENCE_RE = re.compile(r"<evidence>(.*?)</evidence>", re.DOTALL)
_LLM_RE = re.compile(r"<llm>(.*?)</llm>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_response(text: str) -> ParsedResponse:
    evidence_matches = _EVIDENCE_RE.findall(text)
    llm_matches = _LLM_RE.findall(text)
    answer_match = _ANSWER_RE.search(text)

    has_evidence = len(evidence_matches) > 0
    has_llm = len(llm_matches) > 0
    has_answer = answer_match is not None

    one_path = has_evidence ^ has_llm
    format_valid = one_path and has_answer

    if not format_valid:
        return ParsedResponse(path="invalid", raw=text)

    answer = answer_match.group(1).strip()
    if has_evidence:
        return ParsedResponse(
            path="evidence",
            quoted_evidence=[q.strip() for q in evidence_matches],
            answer=answer,
            format_valid=True,
            raw=text,
        )
    return ParsedResponse(
        path="llm",
        answer=answer,
        format_valid=True,
        raw=text,
    )
