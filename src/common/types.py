"""Shared types used by every lane. Imported from rewards, eval, training."""
from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class Sample:
    sample_id: str
    question: str
    gold_passages: list[str]
    distractor_passages: list[str]
    gold_answer: str
    gold_evidence: list[str]
    retrieved_passages: list[str]
    is_answerable: bool


ResponsePath = Literal["evidence", "llm", "invalid"]


@dataclass(slots=True)
class ParsedResponse:
    path: ResponsePath
    quoted_evidence: list[str] = field(default_factory=list)
    answer: str = ""
    format_valid: bool = False
    raw: str = ""
