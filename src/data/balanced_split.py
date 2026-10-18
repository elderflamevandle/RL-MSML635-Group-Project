"""Build answerable/unanswerable split based on retriever output.

For each sample: retrieve k passages. Mark `is_answerable = True` if any
gold passage is in the retrieved set, else `False`.
"""
from typing import Callable
from src.common.types import Sample

RetrieveFn = Callable[[str, int], list[str]]

def build_balanced_split(
    samples: list[Sample], retrieve_fn: RetrieveFn, k: int = 5
) -> list[Sample]:
    out = []
    for s in samples:
        retrieved = retrieve_fn(s.question, k)
        is_ans = any(gp in retrieved for gp in s.gold_passages)
        out.append(
            Sample(
                sample_id=s.sample_id,
                question=s.question,
                gold_passages=s.gold_passages,
                distractor_passages=s.distractor_passages,
                gold_answer=s.gold_answer,
                gold_evidence=s.gold_evidence,
                retrieved_passages=retrieved,
                is_answerable=is_ans,
            )
        )
    return out
