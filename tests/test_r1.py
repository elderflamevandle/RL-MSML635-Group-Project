import pytest
from src.common.types import Sample
from src.rewards.r1_cost_sensitive import r1_cost_sensitive, R1Config

CFG = R1Config()  # defaults from docs/rewards.md


def _ans(**k):
    base = dict(
        sample_id="s", question="Who directed Inception?",
        gold_passages=["Nolan directed Inception."],
        distractor_passages=["x"],
        gold_answer="Christopher Nolan",
        gold_evidence=["Nolan directed Inception."],
        retrieved_passages=["Nolan directed Inception.", "x"],
        is_answerable=True,
    )
    base.update(k)
    return Sample(**base)


def _unans(**k):
    return _ans(gold_passages=[], is_answerable=False,
                retrieved_passages=["x", "y", "z"], **k)


def test_r1_hallucination_scores_negative_lambda():
    # Answered on unanswerable → hallucination.
    resp = "<evidence>Fabricated.</evidence><answer>Christopher Nolan</answer>"
    r, _ = r1_cost_sensitive(resp, _unans(), CFG)
    assert r == -CFG.lam


def test_r1_invalid_format_scores_zero():
    r, _ = r1_cost_sensitive("junk", _ans(), CFG)
    assert r == 0.0


def test_r1_wrong_path_nonhallucination_scores_rf():
    # Answerable, model abstains. Not a hallucination — wrong path only.
    resp = "<llm>no</llm><answer>unknown</answer>"
    r, _ = r1_cost_sensitive(resp, _ans(), CFG)
    assert r == CFG.Rf


def test_r1_correct_llm_path_includes_disclaimer_quality():
    resp = "<llm>I cannot answer from the provided evidence.</llm><answer>unknown</answer>"
    r, parts = r1_cost_sensitive(resp, _unans(), CFG)
    assert r >= CFG.Rf + CFG.Rp  # at least floor
    assert "rouge_l_answer" in parts


def test_r1_evidence_path_quote_in_retrieved_full_credit():
    resp = "<evidence>Nolan directed Inception.</evidence><answer>Christopher Nolan</answer>"
    r, parts = r1_cost_sensitive(resp, _ans(), CFG)
    assert parts["quote_grounded"] is True
    assert r > CFG.Rf + CFG.Rp


def test_r1_evidence_path_fabricated_quote_loses_rf_gate():
    # Quote not in retrieved passages → Rf gate drops to 0.
    resp = "<evidence>This string is not in the passages.</evidence><answer>Christopher Nolan</answer>"
    r, parts = r1_cost_sensitive(resp, _ans(), CFG)
    assert parts["quote_grounded"] is False
    # No Rf; still gets Rp + α·low_rouge + β·high_rouge
    assert 0 < r < CFG.Rf + CFG.Rp + CFG.alpha + CFG.beta
