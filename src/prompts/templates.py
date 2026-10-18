"""Prompt templates. SINGLE SOURCE OF TRUTH for prompt format — both rewards
and training import from here. Do not inline prompt strings elsewhere."""

RAG_SYSTEM = (
    "You are a careful RAG assistant. Read the passages. "
    "If the passages contain the answer, respond with "
    "<evidence>exact quoted sentence(s)</evidence>"
    "<answer>short answer</answer>. "
    "If the passages do NOT contain the answer, respond with "
    "<llm>I cannot answer from the provided evidence.</llm>"
    "<answer>unknown</answer>."
)


def build_prompt(question: str, passages: list[str]) -> str:
    numbered = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    return (
        f"{RAG_SYSTEM}\n\n"
        f"Passages:\n{numbered}\n\n"
        f"Question: {question}\n\n"
        f"Response:"
    )
