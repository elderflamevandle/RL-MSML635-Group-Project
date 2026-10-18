"""HotpotQA loader. Converts raw HF rows into `Sample` objects.

In Week 1 this just maps raw fields through. The balanced ans/unans split and
retrieval are added in Week 2 by `src/data/balanced_split.py` and
`src/data/retrievers.py`.
"""
from pathlib import Path
import json
from src.common.types import Sample


def _hotpot_row_to_sample(row: dict) -> Sample:
    titles = row["context"]["title"]
    sentences = row["context"]["sentences"]
    passages = [" ".join(s) for s in sentences]

    supporting_titles = set(row["supporting_facts"]["title"])
    gold_passages = [
        passages[i] for i, t in enumerate(titles) if t in supporting_titles
    ]
    assert len(gold_passages) > 0, f"No gold passages matched for sample {row['id']}"
    distractor_passages = [
        passages[i] for i, t in enumerate(titles) if t not in supporting_titles
    ]
    gold_evidence = []
    for title, sent_idx in zip(
        row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]
    ):
        if title in titles:
            para_idx = titles.index(title)
            if sent_idx < len(sentences[para_idx]):
                gold_evidence.append(sentences[para_idx][sent_idx])

    return Sample(
        sample_id=row["id"],
        question=row["question"],
        gold_passages=gold_passages,
        distractor_passages=distractor_passages,
        gold_answer=row["answer"],
        gold_evidence=gold_evidence,
        retrieved_passages=gold_passages + distractor_passages,
        is_answerable=True,  # TODO(week-2): balanced_split sets is_answerable=False for mutated samples
    )


def load_hotpot_tiny(path: str | Path) -> list[Sample]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [_hotpot_row_to_sample(json.loads(line)) for line in f]
