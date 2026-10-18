"""Disk cache for retrieval results. Key by (retriever_name, question, k)."""
from pathlib import Path
import json
import hashlib

def _key(retriever_name: str, question: str, k: int) -> str:
    h = hashlib.sha256(f"{retriever_name}|{question}|{k}".encode()).hexdigest()[:16]
    return h

class RetrievalCache:
    def __init__(self, dir: Path):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, retriever_name: str, question: str, k: int) -> list[str] | None:
        p = self.dir / f"{_key(retriever_name, question, k)}.json"
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        if (data["retriever_name"], data["question"], data["k"]) != (retriever_name, question, k):
            raise RuntimeError(
                f"Cache key collision at {p.name}: stored key "
                f"({data['retriever_name']!r}, {data['question']!r}, {data['k']}) "
                f"doesn't match requested ({retriever_name!r}, {question!r}, {k})"
            )
        return data["passages"]

    def set(self, retriever_name: str, question: str, k: int, passages: list[str]) -> None:
        p = self.dir / f"{_key(retriever_name, question, k)}.json"
        payload = {
            "retriever_name": retriever_name,
            "question": question,
            "k": k,
            "passages": passages,
        }
        # Atomic write: temp file then rename.
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(p)
