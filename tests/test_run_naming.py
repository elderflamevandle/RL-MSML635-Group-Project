import time
from src.training.run_naming import make_run_id

def test_run_id_format():
    rid = make_run_id(reward="r0", optimizer="grpo", model="qwen3-0.6b", seed=0)
    # Expect: r0_grpo_qwen3-0.6b_s0_<timestamp>
    parts = rid.split("_")
    assert parts[0] == "r0"
    assert parts[1] == "grpo"
    assert parts[2] == "qwen3-0.6b"
    assert parts[3] == "s0"
    assert len(parts[4]) == 15  # YYYYMMDD-HHMMSS

def test_run_id_unique_across_calls():
    a = make_run_id("r0", "grpo", "qwen3-0.6b", 0)
    time.sleep(1.1)
    b = make_run_id("r0", "grpo", "qwen3-0.6b", 0)
    assert a != b
