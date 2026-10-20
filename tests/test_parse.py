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


# ---- Round-trip contract between build_prompt and parse_response ----
# These lock in the invariant that the response format described in the
# system prompt is exactly what parse_response accepts. If either side drifts,
# these fail — the single-source-of-truth contract is broken.

def _responses_from_system_prompt():
    """The two example responses literally shown in RAG_SYSTEM."""
    from src.prompts.templates import RAG_SYSTEM
    # Sanity: both exemplars must appear in the prompt we show the model.
    assert "<evidence>exact quoted sentence(s)</evidence><answer>short answer</answer>" in RAG_SYSTEM
    assert "<llm>I cannot answer from the provided evidence.</llm><answer>unknown</answer>" in RAG_SYSTEM
    return (
        "<evidence>exact quoted sentence(s)</evidence><answer>short answer</answer>",
        "<llm>I cannot answer from the provided evidence.</llm><answer>unknown</answer>",
    )


def test_prompt_evidence_exemplar_parses_as_evidence():
    evidence_resp, _ = _responses_from_system_prompt()
    r = parse_response(evidence_resp)
    assert r.format_valid is True
    assert r.path == "evidence"
    assert r.quoted_evidence == ["exact quoted sentence(s)"]
    assert r.answer == "short answer"


def test_prompt_llm_exemplar_parses_as_llm():
    _, llm_resp = _responses_from_system_prompt()
    r = parse_response(llm_resp)
    assert r.format_valid is True
    assert r.path == "llm"
    assert r.answer == "unknown"


def test_build_prompt_contains_question_and_passages():
    from src.prompts.templates import build_prompt
    p = build_prompt("Who wrote Dune?", ["Frank Herbert wrote Dune.", "Dune is a sci-fi novel."])
    assert "Who wrote Dune?" in p
    assert "[1] Frank Herbert wrote Dune." in p
    assert "[2] Dune is a sci-fi novel." in p
