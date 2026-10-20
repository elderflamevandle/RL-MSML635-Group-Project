"""QASPER loader. Converts raw HF rows into `Sample` objects.

QASPER (allenai/qasper) is scientific-paper QA. Row shape:
    {
      "id": "<paper id>",
      "title": "...", "abstract": "...",
      "full_text": {"paragraphs": [[str, ...], ...], "section_name": [str, ...]},
      "qas": {
        "question": [str, ...],
        "question_id": [str, ...],
        "answers": [
          {"answer": [{
              "unanswerable": bool,
              "extractive_spans": [str, ...],
              "free_form_answer": str,
              "yes_no": bool | None,
              "evidence": [str, ...],
              "highlighted_evidence": [str, ...],
          }, ...]}, ...
        ],
      },
    }

One paper → N samples (one per question). This loader is minimal; the Week-5
QASPER retrieval build is where ratios + retriever integration happen.
"""
from __future__ import annotations
from pathlib import Path
import json
from src.common.types import Sample


def _flatten_paragraphs(full_text: dict) -> list[str]:
    """Flatten full_text.paragraphs (list[list[str]]) into a single list[str]."""
    out: list[str] = []
    for section in full_text.get("paragraphs", []):
        for para in section:
            if para and para.strip():
                out.append(para)
    return out


def _pick_answer(annotator_answers: list[dict]) -> tuple[str, bool, list[str], list[str]]:
    """Pick the first non-unanswerable annotator's answer.

    Returns (gold_answer, is_answerable, evidence, highlighted_evidence).
    Falls back to the first annotator if all mark unanswerable.
    """
    chosen = None
    for a in annotator_answers:
        if not a.get("unanswerable", False):
            chosen = a
            break
    if chosen is None:
        a0 = annotator_answers[0] if annotator_answers else {}
        return "unknown", False, a0.get("evidence", []) or [], a0.get("highlighted_evidence", []) or []

    # Prefer extractive_spans, then free_form_answer, then yes_no.
    spans = chosen.get("extractive_spans") or []
    free_form = chosen.get("free_form_answer") or ""
    yn = chosen.get("yes_no")
    if spans:
        answer = " ".join(spans).strip()
    elif free_form:
        answer = free_form.strip()
    elif isinstance(yn, bool):
        answer = "yes" if yn else "no"
    else:
        answer = ""
    return (
        answer,
        True,
        chosen.get("evidence", []) or [],
        chosen.get("highlighted_evidence", []) or [],
    )


def _qasper_row_to_samples(row: dict) -> list[Sample]:
    paragraphs = _flatten_paragraphs(row.get("full_text", {}))
    qas = row.get("qas", {})
    questions = qas.get("question", []) or []
    question_ids = qas.get("question_id", []) or []
    answer_blocks = qas.get("answers", []) or []

    samples: list[Sample] = []
    for i, question in enumerate(questions):
        qid = question_ids[i] if i < len(question_ids) else f"{row.get('id', 'q')}_{i}"
        annotator_answers_wrap = answer_blocks[i] if i < len(answer_blocks) else {}
        annotator_answers = annotator_answers_wrap.get("answer", []) or []

        gold_answer, is_answerable, evidence, highlighted = _pick_answer(annotator_answers)

        # Gold passages = paragraphs that match any evidence string (substring).
        # Non-match "FLOAT SELECTED" evidence (tables/figures) is ignored silently.
        gold_passages = [
            p for p in paragraphs
            if any(e.strip() and e.strip() in p for e in evidence)
        ]
        distractor_passages = [p for p in paragraphs if p not in gold_passages]

        samples.append(Sample(
            sample_id=qid,
            question=question,
            gold_passages=gold_passages,
            distractor_passages=distractor_passages,
            gold_answer=gold_answer,
            gold_evidence=list(highlighted),
            retrieved_passages=gold_passages + distractor_passages,
            is_answerable=is_answerable,
        ))
    return samples


def load_qasper(path: str | Path) -> list[Sample]:
    """Load QASPER from a jsonl file where each line is one raw HF row."""
    path = Path(path)
    out: list[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.extend(_qasper_row_to_samples(json.loads(line)))
    return out
