from src.common.parse import parse_response


def test_parse_evidence_path():
    r = parse_response(
        "<evidence>Nolan directed Inception.</evidence><answer>Christopher Nolan</answer>"
    )
    assert r.path == "evidence"
    assert r.format_valid is True
    assert r.quoted_evidence == ["Nolan directed Inception."]
    assert r.answer == "Christopher Nolan"


def test_parse_llm_path():
    r = parse_response("<llm>Not in evidence.</llm><answer>unknown</answer>")
    assert r.path == "llm"
    assert r.format_valid is True
    assert r.answer == "unknown"
    assert r.quoted_evidence == []


def test_parse_invalid_both_paths():
    r = parse_response("<evidence>x</evidence><llm>y</llm><answer>z</answer>")
    assert r.path == "invalid"
    assert r.format_valid is False


def test_parse_invalid_missing_answer():
    r = parse_response("<evidence>x</evidence>")
    assert r.path == "invalid"
    assert r.format_valid is False


def test_parse_multiple_evidence_quotes():
    r = parse_response(
        "<evidence>Sentence one.</evidence>"
        "<evidence>Sentence two.</evidence>"
        "<answer>A</answer>"
    )
    assert r.path == "evidence"
    assert r.quoted_evidence == ["Sentence one.", "Sentence two."]
    assert r.format_valid is True
