"""Unit tests for scripts/score_predictions.py."""
import importlib.util
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "score_predictions_mod", _root / "scripts" / "score_predictions.py"
)
score_mod = importlib.util.module_from_spec(_spec)
sys.modules["score_predictions_mod"] = score_mod
_spec.loader.exec_module(score_mod)
score_predictions = score_mod.score_predictions


def _write_tiny_split(path: Path):
    """Write a tiny HotpotQA-shaped file with 2 samples."""
    rows = [
        {
            "id": "A",
            "question": "Who directed Big Stone Gap?",
            "answer": "Adriana Trigiani",
            "type": "bridge",
            "level": "easy",
            "supporting_facts": {"title": ["BSG"], "sent_id": [0]},
            "context": {
                "title": ["BSG", "Decoy"],
                "sentences": [
                    ["Big Stone Gap was directed by Adriana Trigiani."],
                    ["This passage is irrelevant."],
                ],
            },
        },
        {
            "id": "B",
            "question": "What color is the moon tonight?",
            "answer": "white",
            "type": "bridge",
            "level": "easy",
            "supporting_facts": {"title": ["Moon"], "sent_id": [0]},
            "context": {
                "title": ["Moon", "Decoy"],
                "sentences": [
                    ["The moon tonight is white."],
                    ["Something else entirely."],
                ],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _write_predictions(path: Path):
    """Both samples get the correct <evidence>-path response."""
    rows = [
        {
            "sample_id": "A",
            "is_answerable": True,
            "path": "evidence",
            "answer": "Adriana Trigiani",
            "em": 1.0,
            "raw_response": "<evidence>Big Stone Gap was directed by Adriana Trigiani.</evidence><answer>Adriana Trigiani</answer>",
        },
        {
            "sample_id": "B",
            "is_answerable": True,
            "path": "invalid",
            "answer": "",
            "em": 0.0,
            "raw_response": "Garbage output with no tags.",
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_score_predictions_end_to_end(tmp_path: Path):
    split = tmp_path / "split.jsonl"
    _write_tiny_split(split)
    preds = tmp_path / "predictions.jsonl"
    _write_predictions(preds)
    out = tmp_path / "out"

    summary = score_predictions(preds, split, out)

    assert summary["n_predictions"] == 2
    assert summary["n_scored"] == 2
    assert summary["n_missing_sample"] == 0

    # R0 should give a non-trivial score to the correct response and 0 to garbage
    r0_summary = summary["rewards"]["r0"]
    assert r0_summary["max"] > 1.0  # correct <evidence> path hits >R_f+R_p
    assert r0_summary["min"] == 0.0  # invalid format → 0

    # Output files exist
    assert (out / "rewards.json").exists()
    assert (out / "rewards_per_sample.jsonl").exists()
    assert (out / "reward_hist.png").exists()
    assert (out / "reward_hist.png").stat().st_size > 1000

    # Per-sample file has one row per prediction
    lines = (out / "rewards_per_sample.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    r0_scores = {json.loads(l)["sample_id"]: json.loads(l)["r0"] for l in lines}
    assert r0_scores["A"] > 1.0
    assert r0_scores["B"] == 0.0


def test_score_predictions_skips_unknown_sample_id(tmp_path: Path):
    split = tmp_path / "split.jsonl"
    _write_tiny_split(split)
    preds = tmp_path / "predictions.jsonl"
    preds.write_text(json.dumps({
        "sample_id": "DOES_NOT_EXIST",
        "raw_response": "x",
        "is_answerable": True,
        "path": "invalid",
    }), encoding="utf-8")
    out = tmp_path / "out"

    summary = score_predictions(preds, split, out)
    assert summary["n_predictions"] == 1
    assert summary["n_scored"] == 0
    assert summary["n_missing_sample"] == 1
