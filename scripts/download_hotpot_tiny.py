"""Download first 50 HotpotQA samples for the walking skeleton."""
from datasets import load_dataset
from pathlib import Path
import json

OUT = Path("data/raw/hotpot_tiny.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset("hotpot_qa", "distractor", split="validation[:50]")
with OUT.open("w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps(row) + "\n")

print(f"Wrote {OUT} with 50 samples.")
