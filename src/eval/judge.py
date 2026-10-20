"""Groq LLM-as-Judge client with on-disk cache. Uses Llama 3.3 70B at temp=0."""
import os
import json
import hashlib
import re
from pathlib import Path
import requests
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

MODEL = "llama-3.3-70b-versatile"
URL = "https://api.groq.com/openai/v1/chat/completions"

PROMPT = """You are a strict grader. Given a question, a predicted answer, and a reference answer, decide if the prediction is semantically equivalent to the reference.

Question: {q}
Prediction: {pred}
Reference: {ref}

Respond with exactly: SCORE: 1 (if equivalent) or SCORE: 0 (if not). No other text."""


def _hash(parts) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()[:20]


class GroqJudge:
    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None):
        self.api_key = api_key or os.environ["GROQ_API_KEY"]
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / f"{key}.json" if self.cache_dir else None

    def _cache_get(self, key: str) -> float | None:
        p = self._cache_path(key)
        if p and p.exists():
            return json.loads(p.read_text(encoding="utf-8"))["score"]
        return None

    def _cache_set(self, key: str, score: float):
        p = self._cache_path(key)
        if p:
            p.write_text(json.dumps({"score": score}), encoding="utf-8")

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(requests.HTTPError),
    )
    def _call(self, prompt: str) -> str:
        resp = requests.post(
            URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 10,
            },
            timeout=30,
        )
        if resp.status_code == 429:
            raise requests.HTTPError("rate limit")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def score(self, question: str, prediction: str, reference: str) -> float:
        key = _hash([MODEL, question, prediction, reference])
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        text = self._call(PROMPT.format(q=question, pred=prediction, ref=reference))
        m = re.search(r"SCORE:\s*([01])", text)
        score = float(m.group(1)) if m else 0.0
        self._cache_set(key, score)
        return score
