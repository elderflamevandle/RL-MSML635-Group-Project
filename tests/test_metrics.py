from src.rewards.metrics import rouge_l, quote_in_passages


def test_rouge_l_exact_match():
    assert rouge_l("the cat sat", "the cat sat") == 1.0


def test_rouge_l_zero_overlap():
    assert rouge_l("the cat sat", "dogs run fast") < 0.2


def test_rouge_l_partial():
    score = rouge_l("the cat sat on the mat", "cat sat on mat")
    assert 0.5 < score < 1.0


def test_rouge_l_against_multi_reference_max():
    score = rouge_l("the cat", ["dogs run", "the cat sat"])
    assert score > 0.5  # takes the max over references


def test_quote_in_passages_substring_match():
    assert quote_in_passages("Nolan directed", ["Christopher Nolan directed Inception."]) is True


def test_quote_in_passages_missing():
    assert quote_in_passages("Kubrick directed", ["Christopher Nolan directed Inception."]) is False


def test_quote_in_passages_strips_whitespace():
    assert quote_in_passages("  Nolan directed  ", ["Christopher Nolan directed Inception."]) is True
