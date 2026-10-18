from src.eval.metrics import exact_match, abstention_accuracy, balanced_accuracy

def test_em_exact():
    assert exact_match("Christopher Nolan", "Christopher Nolan") == 1.0

def test_em_case_insensitive_and_punct():
    assert exact_match("christopher  nolan.", "Christopher Nolan") == 1.0

def test_em_mismatch():
    assert exact_match("Kubrick", "Christopher Nolan") == 0.0

def test_abstention_accuracy_correct_refusals():
    # (path, is_answerable) tuples
    preds = [
        ("llm", False),
        ("evidence", True),
        ("llm", False),
        ("evidence", False),
    ]
    # Unanswerable denominators = 3; correct abstentions = 2 (first two "llm" on unanswerable)
    assert abstention_accuracy(preds) == 2 / 3

def test_abstention_accuracy_no_unanswerable_returns_zero():
    preds = [("evidence", True), ("evidence", True)]
    assert abstention_accuracy(preds) == 0.0

def test_balanced_accuracy_simple():
    # 50% EM on answerable, 100% abstention on unanswerable -> 0.75
    assert balanced_accuracy(em_answerable=0.5, abstention_unanswerable=1.0) == 0.75
