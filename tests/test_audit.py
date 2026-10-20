from src.rewards.audit import audit_batch


def test_audit_computes_path_shares():
    responses = [
        ("<evidence>q</evidence><answer>a</answer>", "What is X?"),
        ("<evidence>q</evidence><answer>a</answer>", "What is Y?"),
        ("<llm>no</llm><answer>unknown</answer>", "What is Z?"),
        ("broken", "What is W?"),
    ]
    retrieved = [["q"], ["q"], ["x"], ["z"]]
    report = audit_batch(responses, retrieved)
    assert report["path_shares"]["evidence"] == 0.5
    assert report["path_shares"]["llm"] == 0.25
    assert report["path_shares"]["invalid"] == 0.25


def test_audit_flags_fabricated_quotes():
    responses = [
        ("<evidence>fabricated text</evidence><answer>a</answer>", "q"),
        ("<evidence>q</evidence><answer>a</answer>", "q"),
    ]
    retrieved = [["only this"], ["q"]]
    report = audit_batch(responses, retrieved)
    assert report["fabricated_quote_rate"] == 0.5


def test_audit_flags_question_copy():
    responses = [
        ("<evidence>q</evidence><answer>What is X?</answer>", "What is X?"),
        ("<evidence>q</evidence><answer>correct</answer>", "What is Y?"),
    ]
    retrieved = [["q"], ["q"]]
    report = audit_batch(responses, retrieved)
    assert report["question_copy_rate"] == 0.5
