from unittest.mock import patch, MagicMock
from src.eval.judge import GroqJudge


def test_judge_calls_api_and_returns_score():
    judge = GroqJudge(api_key="fake", cache_dir=None)
    fake_response = MagicMock()
    fake_response.json.return_value = {"choices": [{"message": {"content": "SCORE: 1"}}]}
    fake_response.status_code = 200
    with patch("src.eval.judge.requests.post", return_value=fake_response) as post:
        s = judge.score("What is 2+2?", "4", "4")
    assert s == 1.0
    assert post.called


def test_judge_parses_score_from_messy_output():
    judge = GroqJudge(api_key="fake", cache_dir=None)
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "The answer is semantically equivalent. SCORE: 1"}}]
    }
    fake_response.status_code = 200
    with patch("src.eval.judge.requests.post", return_value=fake_response):
        s = judge.score("q", "ans", "ref")
    assert s == 1.0


def test_judge_cache_hit_skips_api(tmp_path):
    judge = GroqJudge(api_key="fake", cache_dir=tmp_path)
    judge._cache_set("key1", 1.0)
    with patch("src.eval.judge.requests.post") as post:
        s = judge._cache_get("key1")
    assert s == 1.0
    assert not post.called


def test_judge_returns_zero_on_missing_score_marker():
    judge = GroqJudge(api_key="fake", cache_dir=None)
    fake_response = MagicMock()
    fake_response.json.return_value = {"choices": [{"message": {"content": "nope"}}]}
    fake_response.status_code = 200
    with patch("src.eval.judge.requests.post", return_value=fake_response):
        s = judge.score("q", "ans", "ref")
    assert s == 0.0


def test_judge_cache_round_trip(tmp_path):
    judge = GroqJudge(api_key="fake", cache_dir=tmp_path)
    fake_response = MagicMock()
    fake_response.json.return_value = {"choices": [{"message": {"content": "SCORE: 1"}}]}
    fake_response.status_code = 200
    with patch("src.eval.judge.requests.post", return_value=fake_response) as post:
        s1 = judge.score("q", "ans", "ref")
        s2 = judge.score("q", "ans", "ref")
    assert s1 == s2 == 1.0
    assert post.call_count == 1  # second call used cache
