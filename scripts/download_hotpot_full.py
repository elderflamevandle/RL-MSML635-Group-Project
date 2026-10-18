"""Download full HotpotQA train (2000) + validation (500) subsets per proposal.
Server-only: run after `datasets` is installed and plenty of disk is available."""
from datasets import load_dataset
from pathlib import Path
import json

if __name__ == "__main__":
    for split, n, out in [
        ("train", 2000, Path("data/raw/hotpot_train.jsonl")),
        ("validation", 500, Path("data/raw/hotpot_val.jsonl")),
    ]:
        out.parent.mkdir(parents=True, exist_ok=True)
        ds = load_dataset("hotpot_qa", "distractor", split=f"{split}[:{n}]")
        with out.open("w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(row) + "\n")
        print(f"{out}: {n} samples")
