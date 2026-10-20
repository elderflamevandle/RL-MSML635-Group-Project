import pytest
from src.common.types import Sample
from src.rewards.r2_hard_gated import r2_hard_gated, R2Config

CFG = R2Config()


def _ans(**k):
    base = dict(
        sample_id="s", question="q",
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


def test_r2_invalid_format_collapses_to_zero():
    r, _ = r2_hard_gated("junk", _ans(), CFG)
    assert r == 0.0


def test_r2_wrong_path_scores_zero_despite_valid_format():
    # Rpath=0 means both w_p*Rpath and w_q*Rpath*Rquality = 0.
    resp = "<llm>no</llm><answer>unknown</answer>"
    r, parts = r2_hard_gated(resp, _ans(), CFG)
    assert r == 0.0
    assert parts["R_format"] == 1
    assert parts["R_path"] == 0


def test_r2_fabricated_quote_collapses_to_zero():
    # Rformat gate also requires every quote ⊂ retrieved.
    resp = "<evidence>Not in passages.</evidence><answer>Christopher Nolan</answer>"
    r, parts = r2_hard_gated(resp, _ans(), CFG)
    assert r == 0.0
    assert parts["R_format"] == 0


def test_r2_correct_evidence_full_credit():
    resp = "<evidence>Nolan directed Inception.</evidence><answer>Christopher Nolan</answer>"
    r, parts = r2_hard_gated(resp, _ans(), CFG)
    # Rformat=1, Rpath=1, Rquality=0.5*1 + 0.5*1 = 1 → r = w_p + w_q = 1.0
    assert r == pytest.approx(CFG.w_p + CFG.w_q)


def test_r2_correct_llm_scores_wp_plus_wq_times_quality():
    resp = "<llm>I cannot answer.</llm><answer>unknown</answer>"
    r, parts = r2_hard_gated(resp, _unans(), CFG)
    assert r >= CFG.w_p  # at least Rpath credit, possibly more from disclaimer quality
