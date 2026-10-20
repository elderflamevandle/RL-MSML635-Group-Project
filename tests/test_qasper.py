"""Tests for QASPER loader. Uses synthetic rows matching the allenai/qasper schema."""
import json
import pytest
from src.data.qasper import (
    _flatten_paragraphs,
    _pick_answer,
    _qasper_row_to_samples,
    load_qasper,
)


def _row(
    *,
    paper_id="paper-1",
    paragraphs=(("Intro para 1.", "Intro para 2."), ("Related work.",)),
    questions=("What did they do?",),
    question_ids=("q-0",),
    answers=None,
):
    if answers is None:
        answers = [{
            "answer": [{
                "unanswerable": False,
                "extractive_spans": ["They did X."],
                "free_form_answer": "",
                "yes_no": None,
                "evidence": ["Intro para 1."],
                "highlighted_evidence": ["Intro para 1."],
            }]
        }]
    return {
        "id": paper_id,
        "title": "T", "abstract": "A",
        "full_text": {
            "paragraphs": [list(p) for p in paragraphs],
            "section_name": ["Intro", "Related"],
        },
        "qas": {
            "question": list(questions),
            "question_id": list(question_ids),
            "answers": answers,
        },
    }


def test_flatten_paragraphs_drops_empty():
    text = {"paragraphs": [["a", "", "  "], ["b"]]}
    assert _flatten_paragraphs(text) == ["a", "b"]


def test_pick_answer_extractive_spans_preferred():
    answer, is_ans, _, _ = _pick_answer([{
        "unanswerable": False,
        "extractive_spans": ["X.", "Y."],
        "free_form_answer": "free",
        "yes_no": None,
        "evidence": [], "highlighted_evidence": [],
    }])
    assert answer == "X. Y."
    assert is_ans is True


def test_pick_answer_falls_back_to_free_form():
    answer, is_ans, _, _ = _pick_answer([{
        "unanswerable": False,
        "extractive_spans": [],
        "free_form_answer": "free text",
        "yes_no": None,
        "evidence": [], "highlighted_evidence": [],
    }])
    assert answer == "free text"
    assert is_ans is True


def test_pick_answer_yes_no_bool():
    for yn, expected in [(True, "yes"), (False, "no")]:
        answer, _, _, _ = _pick_answer([{
            "unanswerable": False,
            "extractive_spans": [], "free_form_answer": "",
            "yes_no": yn, "evidence": [], "highlighted_evidence": [],
        }])
        assert answer == expected


def test_pick_answer_all_unanswerable():
    answer, is_ans, _, _ = _pick_answer([{
        "unanswerable": True,
        "extractive_spans": [], "free_form_answer": "",
        "yes_no": None, "evidence": [], "highlighted_evidence": [],
    }])
    assert is_ans is False
    assert answer == "unknown"


def test_pick_answer_prefers_first_answerable_over_earlier_unanswerable():
    answer, is_ans, _, _ = _pick_answer([
        {"unanswerable": True, "extractive_spans": [], "free_form_answer": "",
         "yes_no": None, "evidence": [], "highlighted_evidence": []},
        {"unanswerable": False, "extractive_spans": ["Z."], "free_form_answer": "",
         "yes_no": None, "evidence": ["ev"], "highlighted_evidence": ["ev"]},
    ])
    assert is_ans is True
    assert answer == "Z."


def test_row_to_samples_splits_gold_vs_distractor():
    samples = _qasper_row_to_samples(_row())
    assert len(samples) == 1
    s = samples[0]
    assert s.sample_id == "q-0"
    assert s.question == "What did they do?"
    assert s.gold_answer == "They did X."
    assert s.gold_passages == ["Intro para 1."]
    assert "Intro para 2." in s.distractor_passages
    assert "Related work." in s.distractor_passages
    assert s.retrieved_passages == s.gold_passages + s.distractor_passages
    assert s.is_answerable is True
    assert s.gold_evidence == ["Intro para 1."]


def test_row_to_samples_unanswerable_question():
    row = _row(answers=[{
        "answer": [{
            "unanswerable": True, "extractive_spans": [], "free_form_answer": "",
            "yes_no": None, "evidence": [], "highlighted_evidence": [],
        }],
    }])
    samples = _qasper_row_to_samples(row)
    assert len(samples) == 1
    assert samples[0].is_answerable is False
    assert samples[0].gold_answer == "unknown"
    assert samples[0].gold_passages == []  # no evidence → no gold


def test_row_to_samples_multiple_questions_per_paper():
    row = _row(
        questions=("Q1?", "Q2?"),
        question_ids=("q-0", "q-1"),
        answers=[
            {"answer": [{"unanswerable": False, "extractive_spans": ["A1."],
                         "free_form_answer": "", "yes_no": None,
                         "evidence": ["Intro para 1."], "highlighted_evidence": []}]},
            {"answer": [{"unanswerable": False, "extractive_spans": ["A2."],
                         "free_form_answer": "", "yes_no": None,
                         "evidence": ["Related work."], "highlighted_evidence": []}]},
        ],
    )
    samples = _qasper_row_to_samples(row)
    assert [s.sample_id for s in samples] == ["q-0", "q-1"]
    assert samples[0].gold_passages == ["Intro para 1."]
    assert samples[1].gold_passages == ["Related work."]


def test_load_qasper_from_jsonl(tmp_path):
    f = tmp_path / "qasper.jsonl"
    rows = [_row(), _row(paper_id="paper-2", questions=("Q?",), question_ids=("q-9",))]
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = load_qasper(f)
    assert len(out) == 2
    assert out[0].sample_id == "q-0"
    assert out[1].sample_id == "q-9"
