import pytest
from src.common.types import Sample
from src.rewards.r0_grace import r0_grace, R0Config


def _sample_ans(**overrides) -> Sample:
    base = dict(
        sample_id="s", question="q",
        gold_passages=["Christopher Nolan directed Inception."],
        distractor_passages=["Kubrick made 2001."],
        gold_answer="Christopher Nolan",
        gold_evidence=["Christopher Nolan directed Inception."],
        retrieved_passages=["Christopher Nolan directed Inception.", "Kubrick made 2001."],
        is_answerable=True,
    )
    base.update(overrides)
    return Sample(**base)


def _sample_unans(**overrides) -> Sample:
    return _sample_ans(
        gold_passages=["Christopher Nolan directed Inception."],  # non-empty — but marked unanswerable
        retrieved_passages=["Kubrick made 2001.", "Spielberg made Jaws."],
        is_answerable=False,
        **overrides,
    )


CFG = R0Config(Rf=0.5, Rp=1.0, alpha=0.5, beta=0.5)


def test_r0_invalid_format_scores_zero():
    r, parts = r0_grace("no tags here", _sample_ans(), CFG)
    assert r == 0.0
    assert parts["format_valid"] is False


def test_r0_valid_format_wrong_path_scores_rf_only():
    # Answerable sample but model abstains → wrong path.
    resp = "<llm>cannot answer</llm><answer>unknown</answer>"
    r, parts = r0_grace(resp, _sample_ans(), CFG)
    assert r == 0.5
    assert parts["correct_path"] is False


def test_r0_correct_llm_path_scores_rf_plus_rp():
    # Unanswerable sample, model abstains → correct <llm>.
    resp = "<llm>cannot answer</llm><answer>unknown</answer>"
    r, parts = r0_grace(resp, _sample_unans(), CFG)
    assert r == 1.5
    assert parts["correct_path"] is True
    assert parts["rouge_l_evidence"] == 0.0
    assert parts["rouge_l_answer"] == 0.0


def test_r0_correct_evidence_path_full_credit():
    resp = (
        "<evidence>Christopher Nolan directed Inception.</evidence>"
        "<answer>Christopher Nolan</answer>"
    )
    r, parts = r0_grace(resp, _sample_ans(), CFG)
    assert r == 0.5 + 1.0 + 0.5 * 1.0 + 0.5 * 1.0


def test_r0_correct_evidence_path_wrong_quote_partial_credit():
    resp = (
        "<evidence>Something irrelevant about Jaws.</evidence>"
        "<answer>Christopher Nolan</answer>"
    )
    r, parts = r0_grace(resp, _sample_ans(), CFG)
    # Rf + Rp + α·low(RLev) + β·1.0 (answer matches)
    assert 1.5 < r < 2.1


def test_r0_hallucination_still_scores_rf_only():
    # GRACE R0 does NOT penalize hallucination — a key weakness R1 fixes.
    # Unanswerable + evidence path = wrong path for R0.
    resp = (
        "<evidence>Fabricated support text.</evidence>"
        "<answer>Christopher Nolan</answer>"
    )
    r, parts = r0_grace(resp, _sample_unans(), CFG)
    assert r == 0.5  # only format bonus, no path reward


def test_r0_question_copy_still_scores_evidence_credit():
    # If the model copies the question into the answer, R0 doesn't notice.
    # Documented weakness — audit will catch this at eval time.
    s = _sample_ans(gold_answer="Christopher Nolan")
    resp = (
        "<evidence>Christopher Nolan directed Inception.</evidence>"
        "<answer>Who directed Inception?</answer>"
    )
    r, parts = r0_grace(resp, s, CFG)
    # Low RLans but still gets Rf + Rp + α·high + β·low
    assert r > 1.5

def test_r0_zero_overlap_evidence():
    s = _sample_ans()
    resp = (
        "<evidence>Completely unrelated text about dolphins.</evidence>"
        "<answer>Christopher Nolan</answer>"
    )
    r, parts = r0_grace(resp, s, CFG)
    assert parts["rouge_l_evidence"] < 0.2
    assert parts["rouge_l_answer"] == pytest.approx(1.0)

def test_r0_llm_path_on_answerable_still_wrong_path():
    s = _sample_ans()
    resp = "<llm>cannot answer</llm><answer>unknown</answer>"
    r, parts = r0_grace(resp, s, CFG)
    assert r == 0.5
    assert parts["correct_path"] is False
